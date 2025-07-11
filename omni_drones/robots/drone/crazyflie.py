# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import torch

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.robots import ASSET_PATH
from omni_drones.robots.drone import MultirotorBase

import torch.nn as nn


class Crazyflie(MultirotorBase):

    # NOTE: there are unexpedted behaviors when using the asset from Isaac Sim
    usd_path: str = ASSET_PATH + "/usd/cf2x_pybullet.usd"
    # usd_path: str = ASSET_PATH + "/usd/cf2x_isaac.usd"
    param_path: str = ASSET_PATH + "/usd/crazyflie.yaml"

    def initialize(
        self, prim_paths_expr: str = None, track_contact_forces: bool = False
    ):
        super().initialize(
            prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces
        )
        # rotor update
        rotor_config = self.params["rotor_configuration"]
        self.rotors = CFRotor(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = (
            torch.as_tensor(rotor_config["max_rotation_velocities"])
            .float()
            .to(self.device)
        )
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]
        self.throttle_difference = torch.zeros(
            self.throttle.shape[:-1], device=self.device
        )
        logging.info("Use crazyflie rotor")

        # mass & inertia load
        mass = torch.tensor(self.params["mass"])
        inertia = self.params["inertia"]
        inertia = torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])

        # mass & inertia update
        self.base_link.set_masses(mass)
        self.base_link.set_inertias(torch.diag_embed(inertia))

        # update parameters
        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = (
            self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        )
        # default/initial parameters
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link.get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81)  # TODO: get the real g
        self.FORCE2MOMENT_0 = torch.broadcast_to(
            self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape
        )

        logging.info("mass & inertia updated from yaml")
        logging.info(str(self))


class CFRotor(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        tau_up = 0.0125  # crazyflie
        tau_down = 0.025  # crazyflie
        self.num_rotors = len(force_constants)

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        self.KF = nn.Parameter(
            max_rot_vels.square() * force_constants
        )  # max thrust per rotor
        self.KM = nn.Parameter(
            max_rot_vels.square() * moment_constants
        )  # max torque per rotor
        self.throttle = nn.Parameter(
            torch.zeros(self.num_rotors)
        )  # normalized rotor vel [0 ,1]
        self.directions = nn.Parameter(
            torch.as_tensor(rotor_config["directions"]).float()
        )

        self.tau_up = nn.Parameter(tau_up * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(tau_down * torch.ones(self.num_rotors))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor):
        """
        cmds : (N, 4), N is number of drones
        """
        target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))

        # update
        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        decay = torch.exp(-self.dt / tau)
        self.throttle.mul_(decay).add_((1 - decay) * target_throttle)

        noise = torch.randn_like(self.throttle) * self.noise_scale * 0.0
        t = torch.clamp(self.f(self.throttle) + noise, 0.0, 1.0)
        thrusts = t * self.KF
        moments = (t * self.KM) * -self.directions

        return thrusts, moments
