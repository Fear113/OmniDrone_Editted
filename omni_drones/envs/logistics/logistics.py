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
import copy
import dataclasses

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch
import torch.distributions as D
import numpy as np

from omni_drones.envs.logistics.utils import StateSnapshot, ConnectedPayload, DisconnectedPayload, Group
from omni_drones.views import RigidPrimView

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.envs.transport.utils import TransportationCfg, TransportationGroup
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.utils.torch import (
    normalize, quat_rotate, quat_rotate_inverse, quat_axis, symlog
)

from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
from omni.kit.commands import execute


class Logistics(IsaacEnv):
    def __init__(self, cfg, headless, initial_state: Optional[StateSnapshot] = None):
        self.device = torch.device("cuda:0")
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.num_groups = cfg.task.num_groups
        self.num_payloads_per_group = cfg.task.num_payloads_per_group
        self.num_drones_per_group = cfg.task.num_drones_per_group
        self.group=None
        self.group_offset = self.make_group_offset()
        self.payload_offset = self.make_payload_offset()
        self.initial_state = initial_state if initial_state is not None else self.make_initial_state()
        self.done_group = None


        super().__init__(cfg, headless)

        self.drone.initialize(f"/World/envs/env_*/Group_*/{self.drone.name.lower()}_*")
        self.alpha = 0.8
        self.count = [0 for _ in range(self.num_groups)]
        self.world = World()
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0.sum(), device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0.sum(), device=self.device)
        )
        if not self.group is None:
            self.group.initialize()

    def snapshot_state(self):
        drone_state = self.drone.get_state()
        groups = []

        for i, group in enumerate(self.initial_state.groups):
            drone_range = range(i * self.num_drones_per_group,
                                i * self.num_drones_per_group + self.num_drones_per_group)
            drone_pos = drone_state[..., drone_range, 0:3]
            drone_rot = drone_state[..., drone_range, 3:7]
            drone_vel = drone_state[..., drone_range, 7:13]

            if self.done_group == i:
                is_transporting = not group.is_transporting
                if group.is_transporting:
                    target_payload_idx = group.target_payload_idx + 1 if group.target_payload_idx < self.num_payloads_per_group else None
                else:
                    target_payload_idx = group.target_payload_idx

                payloads = []
                for j, payload in enumerate(group.payloads):
                    if target_payload_idx == j and group.is_transporting:
                        _payload = DisconnectedPayload(
                            payload.target_pos,
                            payload.target_rot,
                            payload.payload_pos,
                            payload.payload_rot
                        )
                        payloads.append(_payload)
                    elif target_payload_idx == j and not group.is_transporting:
                        _payload = ConnectedPayload(
                            payload.target_pos,
                            payload.target_rot,
                            payload.payload_pos,
                            payload.payload_rot,
                            torch.zeros((1, 32)),
                            torch.zeros((1, 32)),
                        )
                        payloads.append(_payload)
                    else:
                        payloads.append(payload)
            else:
                is_transporting = group.is_transporting
                target_payload_idx = group.target_payload_idx
                payloads = group.payloads

            group = Group(
                drone_pos,
                drone_rot,
                drone_vel,
                target_payload_idx,
                is_transporting,
                payloads
            )

            groups.append(group)

        return StateSnapshot(groups)

    def make_group_offset(self):
        group_interval = 3
        group_offset = torch.zeros(self.num_groups, 3, device=self.device)
        group_offset[:, 0] = torch.arange(start=0, end=group_interval * self.num_groups, step=group_interval,
                                          device=self.device)

        return group_offset
    
    def make_payload_offset(self):
        payload_offset = []
        for i in range(self.num_payloads_per_group):
            payload_position = [0, -2+i*2, 0]
            payload_offset.append(payload_position)
        return torch.FloatTensor(payload_offset).to(device=self.device)

    def make_initial_state(self):
        drone_formation = torch.tensor([
            [1, 1, 1.5],
            [1, -1, 1.5],
            [-1, 1, 1.5],
            [-1, -1, 1.5]
        ], device=self.device)
        self.formation = drone_formation

        payload_pos_dist = D.Uniform(
            torch.tensor([-1., -1., 0.25], device=self.device),
            torch.tensor([1., 1., 0.25], device=self.device)
        )
        payload_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )

        groups = []
        for i in range(self.num_groups):
            drone_pos = drone_formation + self.group_offset[i]
            drone_rot = torch.zeros((self.num_drones_per_group,4), device=self.device)
            drone_rot[:,0] = 1
            drone_vel = torch.zeros((self.num_drones_per_group,6), device=self.device)
            target_payload_idx = 0
            is_transporting = False
            payloads = []

            for j in range(self.num_payloads_per_group):
                payload_target_pos = self.group_offset[i] + torch.tensor([0., 2., j * 2], device=self.device)
                payload_target_rot = torch.zeros(4, device=self.device)
                # payload_pos = payload_pos_dist.sample() + self.group_offset[i]
                payload_pos = payload_pos_dist.sample() + self.group_offset[i] + self.payload_offset[j]
                payload_rot = euler_to_quaternion(payload_rpy_dist.sample())
                payloads.append(DisconnectedPayload(payload_target_pos, payload_target_rot, payload_pos, payload_rot))

            groups.append(
                Group(drone_pos, drone_rot, drone_vel, target_payload_idx, is_transporting, payloads)
            )
            

        return StateSnapshot(groups)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        scene_utils.design_scene()

        for i, group in enumerate(self.initial_state.groups):
            group_prim_path = f"/World/envs/env_0/Group_{i}"
            # spawn drones
            if group.is_transporting:
                group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones_per_group)
                self.group = TransportationGroup(drone=self.drone, cfg=group_cfg)
                #payload_location = drone_pos.mean(axis=0)
                payload_position = group.payloads[group.target_payload_idx].payload_pos.clone().detach()
                # payload_position[2] =0.25
                self.group.spawn(translations=payload_position, enable_collision=True)  #drone_translations_origin=group.drone_pos.clone().detach()
            else:
                prim_utils.create_prim(group_prim_path)  # xform
                drone_prim_paths = [f"{group_prim_path}/{self.drone.name.lower()}_{j}" for j in
                                    range(self.num_drones_per_group)]
                self.drone.spawn(translations=group.drone_pos, prim_paths=drone_prim_paths)

            # spawn payload
            for j, payload in enumerate(group.payloads):
                if isinstance(payload, DisconnectedPayload):
                    self.create_payload(payload.payload_pos, f"{group_prim_path}/payload_{j}")

    def _reset_idx(self, env_ids: torch.Tensor):
        pos = self.initial_state.groups[0].drone_pos.clone().detach().to(device=self.device).unsqueeze(dim=0)
        rot = self.initial_state.groups[0].drone_rot.clone().detach().to(device=self.device).unsqueeze(dim=0)
        vel = self.initial_state.groups[0].drone_vel.clone().detach().to(device=self.device).unsqueeze(dim=0)
        if not self.initial_state.groups[0].is_transporting:
            self.drone._reset_idx(env_ids)
            self.drone.set_world_poses(pos, rot, env_ids)
            self.drone.set_velocities(vel, env_ids)
        else:
            self.group._reset_idx(env_ids)
            self.group.drone.set_world_poses(pos, rot, env_ids)
            self.group.drone.set_velocities(vel, env_ids)
            # init_joint_pos = self.initial_state.groups[0].payloads[0].joint_pos.clone().detach().to(device=self.device).unsqueeze(dim=0)
            # init_joint_vel = self.initial_state.groups[0].payloads[0].joint_vel.clone().detach().to(device=self.device).unsqueeze(dim=0)

            # self.group.set_joint_positions(init_joint_pos, env_ids)
            # self.group.set_joint_velocities(init_joint_vel, env_ids)
            payload = self.group.payload_view
            payload_masses = self.payload_mass_dist.sample(env_ids.shape)
            payload.set_masses(payload_masses, env_ids)


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n - 1, 13 + 1)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        for i, group in enumerate(self.initial_state.groups):
            if group.is_transporting:
                obs = TensorDict({}, [self.num_envs, self.drone.n])
                state = TensorDict({}, self.batch_size)

                return TensorDict({
                    "agents": {
                        "observation": obs,
                        "observation_central": state,
                    }
                }, self.batch_size)
            else:
                self.root_states = self.drone.get_state()
                pos = self.drone.pos
                payload = group.payloads[group.target_payload_idx]
                target_pos = payload.payload_pos.clone().detach()
                target_pos[2] +=2
                self.root_states[..., :3] = target_pos - pos

                obs_self = [self.root_states]
                if self.time_encoding:
                    t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
                    obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
                obs_self = torch.cat(obs_self, dim=-1)

                relative_pos = torch.vmap(cpos)(pos, pos)
                self.drone_pdist = torch.vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
                relative_pos = torch.vmap(off_diag)(relative_pos)

                obs_others = torch.cat([
                    relative_pos,
                    self.drone_pdist,
                    torch.vmap(others)(self.root_states[..., 3:13])
                ], dim=-1)

                obs = TensorDict({
                    "obs_self": obs_self.unsqueeze(2),
                    "obs_others": obs_others,
                    "pos": self.drone.pos,
                }, [self.num_envs, self.drone.n])

                state = TensorDict({"drones": self.root_states}, self.batch_size)

                return TensorDict({
                    "agents": {
                        "observation": obs,
                        "observation_central": state,
                    }
                }, self.batch_size)

    def _compute_reward_and_done(self):
        for i, group in enumerate(self.initial_state.groups):
            if group.is_transporting:
                payload = group.payloads[group.target_payload_idx]
                pos = self.drone.pos
                distance = torch.norm(pos.mean(-2, keepdim=True) - payload.target_pos, dim=-1)
                truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
                crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)
                if distance < 1:
                    self.count[i] += 1
                terminated = crash | (self.count[i] > 49)
                reward = torch.FloatTensor([[0]]).to(device=self.device)
                
            else:
                payload = group.payloads[group.target_payload_idx]
                pos = self.drone.pos

                cost_h = cost_formation_hausdorff(pos, desired_p=self.formation)
                target_pos = payload.payload_pos.clone().detach()
                target_pos[2] +=2
                distance = torch.norm(pos.mean(-2, keepdim=True) - target_pos, dim=-1)

                reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
                reward_pos = torch.exp(- distance)
                reward_heading = self.drone.heading[..., 0].mean(-1, True)

                separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
                reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
                # reward = (
                #     reward_separation * (
                #         reward_formation 
                #         + reward_formation * (reward_pos + reward_heading)
                #         + 0.4 * reward_pos
                #     )
                # )
                reward = torch.FloatTensor([[0]]).to(device=self.device)
                truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
                crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

                terminated = crash | (distance<0.1)
                

            if terminated | truncated:
                self.done_group = i
                self.count[i] = 0

            return TensorDict(
                {
                    "agents": {
                        "reward": reward.unsqueeze(1).expand(-1, self.drone.n, 1)
                    },
                    "done": terminated | truncated,
                    "terminated": terminated,
                    "truncated": truncated,
                },
                self.batch_size,
            )

    def create_payload(self, pos, prim_path):
        payload = prim_utils.create_prim(
            prim_path=prim_path,
            prim_type="Cube",
            position=pos,
            scale=(0.75, 0.5, 0.2),
        )

        script_utils.setRigidBody(payload, "convexHull", False)
        UsdPhysics.MassAPI.Apply(payload)
        payload.GetAttribute("physics:mass").Set(2.0) # TODO: adjust mass
        payload.GetAttribute("physics:collisionEnabled").Set(True)

        kit_utils.set_rigid_body_properties(
            payload.GetPath(),
            angular_damping=0.1,
            linear_damping=0.1
        )




@torch.vmap
def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    if p.dim()==1:
        p = p.reshape(4,3)
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost.unsqueeze(-1)
def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d