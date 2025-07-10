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

class CFRotor(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        tau_up = 0.0125
        tau_down = 0.025
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
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
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

        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        self.throttle.add_(tau * (target_throttle - self.throttle))

        noise = torch.randn_like(self.throttle) * self.noise_scale * 0.0
        t = torch.clamp(self.f(self.throttle) + noise, 0.0, 1.0)
        thrusts = t * self.KF
        moments = (t * self.KM) * -self.directions

        return thrusts, moments
