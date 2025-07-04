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


import torch

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import DSLPIDController, RateController
from omni_drones.robots import ASSET_PATH
from omni_drones.robots.drone import MultirotorBase

import logging
from typing import Type, Dict

import torch
import torch.distributions as D
import yaml
from functorch import vmap
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

from omni_drones.views import RigidPrimView
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import LeePositionController

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog
)

from dataclasses import dataclass
from collections import defaultdict

import pprint


class Crazyflie(MultirotorBase):
    # NOTE: there are unexpedted behaviors when using the asset from Isaac Sim
    usd_path: str = ASSET_PATH + "/usd/cf2x_pybullet.usd"
    # usd_path: str = ASSET_PATH + "/usd/cf2x_isaac.usd"
    param_path: str = ASSET_PATH + "/usd/crazyflie.yaml"
    # DEFAULT_CONTROLLER = DSLPIDController
    DEFAULT_CONTROLLER = RateController

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:

        '''
        actions : (f, r, p, y)
        1. apply action design -> deok
            acion : f, roll, pitch, yaw
        2. USD information

        Step
        1. Receive apply action
        2. Use the function and train hover
        3. Send trained agent
        4. Verification
            -aciont : f, roll, pitch, yaw

        rotors = Ratecontroller.forward(state, actions[1:], actions[0])
        '''
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        # TODO@btx0424: general rotating rotor
        if self.is_articulation and self.rotor_joint_indices is not None:
            rot_vel = (self.throttle * self.directions * self.MAX_ROT_VEL)
            self._view.set_joint_velocities(
                rot_vel.reshape(-1, self.num_rotors),
                joint_indices=self.rotor_joint_indices
            )
        self.forces.zero_()
        # TODO: global downwash
        if self.n > 1:
            self.forces[:] += vmap(self.downwash)(
                self.pos,
                self.pos,
                quat_rotate(self.rot, self.thrusts.sum(-2)),
                kz=0.3
            ).sum(-2)
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3),
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3),
            self.torques.reshape(-1, 3),
            is_global=True
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)
