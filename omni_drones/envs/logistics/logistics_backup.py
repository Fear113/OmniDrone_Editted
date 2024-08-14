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
from dataclasses import dataclass

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch
import torch.distributions as D
import numpy as np

from omni_drones.envs.logistics.utils import InitialState, ConnectedPayload, DisconnectedPayload
from omni_drones.views import RigidPrimView

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.envs.transport.utils import TransportationCfg, TransportationGroup
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
from omni.kit.commands import execute


def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]


class Logistics(IsaacEnv):
    def __init__(self, cfg, headless, initial_state: Optional[InitialState] = None):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.initial_state = initial_state
        self.num_groups = cfg.task.num_groups
        self.num_drones_per_group = cfg.task.num_drones_per_group
        self.num_payloads_per_group = cfg.task.num_payloads_per_group

        super().__init__(cfg, headless)

        self.drone.initialize(f"/World/envs/env_*/Group_*/{self.drone.name.lower()}_*")
        self.init_poses = self.drone.get_world_poses(clone=True)

        # initial state distribution
        self.cells = (
            make_cells([-2, -2, 0.5], [2, 2, 2], [0.5, 0.5, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )
        # self.init_rpy_dist = D.Uniform(
        #     torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
        #     torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        # )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )
        
        for i, group in enumerate(self.initial_state.groups):
            if group.is_transporting:
                self.target_pos = torch.tensor([0.0, 0.0, 1.5], device=self.device)
                self.target_pos = self.target_pos.expand(self.num_envs, 1, 3)
            else:
                self.target_pos = torch.tensor([
                [0.0,0.0,1.5]
            ], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading[..., 0] = -1

        self.alpha = 0.8
        self.world = World()
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        scene_utils.design_scene()

        # set group position offset
        if self.num_groups == 1:
            group_offset = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        else:
            raise ValueError(f"num_groups: {self.num_groups} not supported")

        # set formation of initial position
        if self.num_drones_per_group == 6:
            formation = torch.tensor([
                [1.7321, -1, 2.0],
                [0, -2, 2.0],
                [-1.7321, -1, 2.0],
                [-1.7321, 1.0, 2.0],
                [0.0, 2.0, 2.0],
                [1.7321, 1.0, 2.0],
            ], device=self.device)
        elif self.num_drones_per_group ==4:
            formation = torch.tensor([
                [1, 1, 0.5],
                [1, -1, 0.5],
                [-1,1, 0.5],
                [-1,-1,0.5]
            ], device=self.device)
        else:
            raise ValueError(f"num_drones_per_group: {self.num_drones_per_group} not supported")
        self.formation = formation.clone()
        # set up initial state
        if self.initial_state is None:
            for i in range(self.num_groups):
                drone_pos = formation + group_offset[i]
                # drone_rot = None # TODO
                drone_rot = torch.tensor([[0,0,0,0]] * self.num_drones_per_group,device=self.device)
                # drone_vel = None # TODO
                drone_vel = torch.tensor([[0,0,0]]* self.num_drones_per_group,device=self.device)
                is_transporting = False
                payloads = []
                for j in range(self.num_payloads_per_group):
                    payload_pos = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)


        for i, group in enumerate(self.initial_state.groups):
            group_prim_path = f"/World/envs/env_0/Group_{i}"
            # spawn drones
            if group.is_transporting:
                group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones_per_group)
                self.group = TransportationGroup(drone=self.drone, cfg=group_cfg)
                # self.group.initialize()
                self.group.spawn(translations=group_offset[i], enable_collision=True, drone_translations_origin=self.initial_state.groups[0].drone_pos)
            else:
                prim_utils.create_prim(group_prim_path)  # xform
                drone_prim_paths = [f"{group_prim_path}/{self.drone.name.lower()}_{j}" for j in
                                    range(self.num_drones_per_group)]
                self.drone.spawn(translations=formation + group_offset[i], prim_paths=drone_prim_paths)

            # self.recover(group.drone_pos, group.drone_rot, group.drone_vel, i)

            # spawn payload
            for j, payload in enumerate(group.payloads):
                if isinstance(payload, DisconnectedPayload):
                    self.create_payload(group_offset[i], f"{group_prim_path}/payload_{j}")

        #
        #
        # # spawn group xform
        # self.xform_prims = []
        # for i in range(self.num_groups):
        #     xform = prim_utils.create_prim(f"/World/envs/env_0/Group_{i}")
        #     self.xform_prims.append(xform)
        #
        # # set group position offset
        # if self.num_groups == 1:
        #     group_offset = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        # else:
        #     raise ValueError(f"num_groups: {self.num_groups} not supported")
        #
        # # spawn drones
        # if self.num_drones_per_group == 6:
        #     formation = torch.tensor([
        #         [1.7321, -1, 1.5],
        #         [0, -2, 1.5],
        #         [-1.7321, -1, 1.5],
        #         [-1.7321, 1.0, 1.5],
        #         [0.0, 2.0, 1.5],
        #         [1.7321, 1.0, 1.5],
        #     ], device=self.device)
        # else:
        #     raise ValueError(f"num_drones_per_group: {self.num_drones_per_group} not supported")
        #
        # drone_prim_paths = []
        # for i in range(self.num_groups):
        #     for j in range(self.num_drones_per_group):
        #         drone_prim_paths.append(f"/World/envs/env_0/Group_{i}/{self.drone.name.lower()}_{j}")
        #
        # drone_offset = group_offset.repeat_interleave(self.num_drones_per_group, dim=0)
        # self.drone_prims = np.array(self.drone.spawn(translations=formation + drone_offset, prim_paths=drone_prim_paths))
        # self.drone_prims = self.drone_prims.reshape(self.num_groups, self.num_drones_per_group)
        #
        # # spawn payloads
        # if self.num_payloads_per_group == 1:
        #     payload_pos = torch.tensor([[0., -2., 0.2]], device=self.device)
        # else:
        #     raise ValueError(f"num_payloads_per_group: {self.num_payloads_per_group} not supported")
        # payload_offset = group_offset.repeat_interleave(self.num_payloads_per_group, dim=0)
        # payload_pos = payload_pos.repeat(self.num_groups, 1) + payload_offset
        # payload_prim_paths = []
        # for i in range(self.num_groups):
        #     for j in range(self.num_payloads_per_group):
        #         payload_prim_paths.append(f"/World/envs/env_0/Group_{i}/payload_{j}")
        #
        # self.payload_prims = np.array(self.create_payload(payload_pos, payload_prim_paths))
        # self.payload_prims = self.payload_prims.reshape(self.num_groups, self.num_payloads_per_group)
        #
        # # link drone and payload if needed
        # for i, group in enumerate(self.last_state.groups):
        #     for j, payload in enumerate(group.payloads):
        #         if isinstance(payload, ConnectedPayload):
        #             for drone_prim in self.drone_prims[i]:
        #                 execute(
        #                     "UnapplyAPISchema",
        #                     api=UsdPhysics.ArticulationRootAPI,
        #                     prim=drone_prim,
        #                 )
        #                 execute(
        #                     "UnapplyAPISchema",
        #                     api=PhysxSchema.PhysxArticulationAPI,
        #                     prim=drone_prim,
        #                 )
        #                 scene_utils.create_bar(
        #                     prim_path=f"{drone_prim.GetPath()}/bar",
        #                     length=1,
        #                     translation=(0, 0, -0.5),
        #                     from_prim=self.payload_prims[i][j],
        #                     to_prim=f"{drone_prim.GetPath()}/base_link",
        #                     mass=0.03,
        #                     enable_collision=True
        #                 )
        #             UsdPhysics.ArticulationRootAPI.Apply(self.xform_prims[i])
        #             PhysxSchema.PhysxArticulationAPI.Apply(self.xform_prims[i])
        #             kit_utils.set_articulation_properties(
        #                 self.xform_prims[i].GetPath(),
        #                 enable_self_collisions=False,
        #                 solver_position_iteration_count=cfg.articulation_props.solver_position_iteration_count,
        #                 solver_velocity_iteration_count=cfg.articulation_props.solver_velocity_iteration_count,
        #             )
        #
        # return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # pos = torch.vmap(sample_from_grid, randomness="different")(
        #     self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        # ) + self.envs_positions[env_ids].unsqueeze(1)
        if self.initial_state.groups[0].drone_pos is None:
            pos = torch.tensor([[[1.0, 1.0, 2.5],
                [1.0, -1.0, 2.5],
                [-1.0, 1.0, 2.5],
                [-1.0, -1.0, 2.5]
                ]], device=self.device)
        else:
            pos = torch.tensor(self.initial_state.groups[0].drone_pos, device=self.device)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)
        if pos.dim()==3:
            temp = torch.vmap(cost_formation_hausdorff)(
            pos.reshape(pos.shape[0],1,12), desired_p=self.formation
            )
            self.last_cost_h[env_ids] = temp.squeeze(1)
        else:
            self.last_cost_h[env_ids] = torch.vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            # "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            # "obs_others": UnboundedContinuousTensorSpec((self.drone.n - 1, 13 + 1)),
            "pos": UnboundedContinuousTensorSpec((self.drone.n, 3)), 
            "rotation": UnboundedContinuousTensorSpec((self.drone.n, 4))
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
                # self.drone_states = self.drone.get_state()
                # self.group.get_state()
                obs = TensorDict({}, [self.num_envs, self.drone.n])
                state = TensorDict({}, self.batch_size)

                return TensorDict({
                    "agents": {
                        "observation": obs, 
                        "state": state,
                    }
                }, self.num_envs)
            else:
                self.root_states = self.drone.get_state()
                pos = self.drone.pos
                self.root_states[..., :3] = self.target_pos - pos

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
                    # "stats": self.stats
                }, self.batch_size)

    def _compute_reward_and_done(self):
        # # placeholder reward and never True done
        # return TensorDict(
        #     {
        #         "agents": {
        #             "reward": torch.tensor([[0]]).unsqueeze(1).expand(-1, self.drone.n, 1)
        #         },
        #         "done": torch.tensor([[False]]),
        #         "terminated": torch.tensor([[False]]),
        #         "truncated": torch.tensor([[False]]),
        #     },
        #     self.batch_size,
        # )
        for i, group in enumerate(self.initial_state.groups):
            if group.is_transporting:
                pass #Todo
            else:
                pos = self.drone.pos

                cost_h = cost_formation_hausdorff(pos, desired_p=self.formation)
                
                distance = torch.norm(pos.mean(-2, keepdim=True) - self.target_pos, dim=-1)

                reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
                # reward_pos = 1 / (1 + cost_pos)

                # reward_formation = torch.exp(- cost_h * 1.6)
                reward_pos = torch.exp(- distance)
                reward_heading = self.drone.heading[..., 0].mean(-1, True)

                separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
                reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
                reward = (
                    reward_separation * (
                        reward_formation 
                        + reward_formation * (reward_pos + reward_heading)
                        + 0.4 * reward_pos
                    )
                )

                # self.last_cost_l[:] = cost_l
                self.last_cost_h[:] = cost_h
                self.last_cost_pos[:] = torch.square(distance)

                truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
                crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

                # terminated = crash | (separation<0.23)
                terminated = crash | (distance<0.23)
                # self.stats["return"].add_(reward)
                # self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
                # # self.stats["cost_laplacian"] -= cost_l
                # self.stats["cost_hausdorff"].lerp_(cost_h, (1-self.alpha))
                # self.stats["pos_error"].lerp_(distance, (1-self.alpha))

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

    # def recover(self, pos, rot, vel, env_ids):
    #     if not(pos is None) or not (rot is None):
    #         self.drone.set_world_poses(pos, rot, env_ids) ######### not env_ids, but group ids should be applied. 
    #     if vel:
    #         self.drone.set_velocities(vel, env_ids)

    def create_payload(self, pos, prim_path):
        payload = prim_utils.create_prim(
            prim_path=prim_path,
            prim_type="Cube",
            translation=pos,
            scale=(0.75, 0.5, 0.2), 
        )

        script_utils.setRigidBody(payload, "convexHull", False)
        UsdPhysics.MassAPI.Apply(payload)
        payload.GetAttribute("physics:mass").Set(2.0)
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
