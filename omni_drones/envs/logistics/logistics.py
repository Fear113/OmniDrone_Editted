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
import random

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch
import torch.distributions as D
import numpy as np

from omni_drones.envs.logistics import state_snapshot
from omni_drones.envs.logistics.state_snapshot import StateSnapshot, ConnectedPayload, DisconnectedPayload, \
    GroupSnapshot, Stage
from omni_drones.utils.payload import Payload
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
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Sdf
from omni.kit.commands import execute
from omni.usd import get_world_transform_matrix
from omni.isaac.core.objects import DynamicSphere
from omni_drones.robots.robot import ASSET_PATH

@dataclass
class Group:
    drones: MultirotorBase
    payloads: []
    transport: Optional[TransportationGroup]

class Logistics(IsaacEnv):
    def __init__(self, cfg, headless, initial_state: Optional[StateSnapshot] = None):
        self.device = torch.device("cuda:0")
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.num_groups = cfg.task.num_groups
        self.num_payloads_per_group = cfg.task.num_payloads_per_group
        self.num_drones_per_group = cfg.task.num_drones_per_group
        self.drone_num = self.num_drones_per_group * self.num_groups
        self.groups = []
        self.enable_background = cfg.task.enable_background
        drone_formation = torch.tensor([
            [-0.5, -0.5, 1.0],
            [-0.5, 0.5, 1.0], #
            [0.5, 0.5, 1.0],
            [0.5, -0.5, 1.0]
        ], device=self.device)
        self.formation = drone_formation
        self.group_offset = self.make_group_offset()
        self.payload_offset = self.make_payload_offset()
        self.initial_state = initial_state if initial_state is not None else self.make_initial_state()
        self.done_group = None
        self.done_payloads = self.initial_state.done_payloads

        super().__init__(cfg, headless)

        for i, group in enumerate(self.groups):
            if group.transport is not None:
                group.transport.initialize(f"/World/envs/env_*/Group_{i}")
            group.drones.initialize(f"/World/envs/env_*/Group_{i}/{self.groups[i].drones.name.lower()}_*")

        self.alpha = 0.8
        self.count = [snapshot.count for snapshot in self.initial_state.group_snapshots]
        self.world = World()

    def snapshot_state(self):
        group_snapshots = []


        for i, group_snapshot in enumerate(self.initial_state.group_snapshots):
            drone_pos = self.groups[i].drones.get_state()[..., 0:3].squeeze(axis=0)
            drone_rot = self.groups[i].drones.get_state()[..., 3:7].squeeze(axis=0)
            drone_vel = self.groups[i].drones.get_state()[..., 7:13].squeeze(axis=0)
            count = self.count[i]

            if self.done_group == i:
                drone_rot = torch.zeros((self.num_drones_per_group, 4), device=self.device)
                drone_rot[:, 0] = 1
                drone_vel = torch.zeros((self.num_drones_per_group, 6), device=self.device)
                count = 0

                stage = group_snapshot.stage.next()
                target_payload_idx = group_snapshot.target_payload_idx
                payloads = []
                for j, payload in enumerate(group_snapshot.payloads):
                    if target_payload_idx == j:
                        if group_snapshot.stage == Stage.FORMATION:
                            tempPayload_path = self.groups[i].payloads[j].GetPath()
                            tempPayload = RigidPrimView(
                                f"{tempPayload_path}",
                                reset_xform_properties=False,
                                track_contact_forces=False
                            )
                            current_payload_pos, current_payload_rot = self.get_env_poses(tempPayload.get_world_poses())

                            _payload = DisconnectedPayload(
                                payload.type,
                                payload.detail().target_pos,
                                current_payload_pos.squeeze(axis=0),
                                current_payload_rot.squeeze(axis=0)
                            )
                            payloads.append(_payload)
                        elif group_snapshot.stage == Stage.POST_FORMATION:
                            world_transform_matrix = get_world_transform_matrix(self.groups[i].payloads[j])
                            temp_pos = world_transform_matrix.ExtractTranslation()
                            temp_pos[2] += 1
                            temp_quatd = world_transform_matrix.ExtractRotationQuat()
                            orient = np.insert(np.array(temp_quatd.imaginary), 0, temp_quatd.real)
                            target_pos = payload.detail().target_pos
                            target_pos = (target_pos[0] + self.done_payloads[payload.detail().name] * 3.0, target_pos[1], target_pos[2])
                            _payload = ConnectedPayload(
                                payload.type,
                                target_pos,
                                torch.FloatTensor(temp_pos).to(device=self.device),
                                torch.FloatTensor(orient).to(device=self.device),
                                torch.zeros((1, 6)),
                                torch.zeros((1, 32)),
                                torch.zeros((1, 32)),
                            )
                            payloads.append(_payload)
                        elif group_snapshot.stage == Stage.PRE_TRANSPORT:
                            pos, rot = self.groups[i].transport.get_world_poses(True)
                            vel = self.groups[i].transport.get_velocities(True)
                            joint_pos = self.groups[i].transport.get_joint_positions(True)
                            joint_vel = self.groups[i].transport.get_joint_velocities(True)
                            target_pos = payload.detail().target_pos
                            target_pos = (target_pos[0] + self.done_payloads[payload.detail().name] * 3.0, target_pos[1], target_pos[2])

                            payloads.append(ConnectedPayload(
                                payload.type,
                                target_pos,
                                pos.squeeze(axis=0),
                                rot.squeeze(axis=0),
                                vel.squeeze(axis=0),
                                joint_pos.squeeze(axis=0),
                                joint_vel.squeeze(axis=0)
                            ))
                        elif group_snapshot.stage == Stage.TRANSPORT:
                            self.done_payloads[payload.detail().name] += 1
                            tempPayload = self.groups[i].transport.payload_view
                            current_payload_pos, current_payload_rot = self.get_env_poses(tempPayload.get_world_poses())

                            drone_rot = torch.zeros((self.num_drones_per_group, 4), device=self.device)
                            drone_rot[:, 0] = 1
                            drone_vel = torch.zeros((self.num_drones_per_group, 6), device=self.device)
                            _payload = DisconnectedPayload(
                                payload.type,
                                payload.detail().target_pos,
                                current_payload_pos.squeeze(axis=0),
                                current_payload_rot.squeeze(axis=0)
                            )
                            payloads.append(_payload)
                        else:
                            raise NotImplementedError
                    else:
                        tempPayload_path = self.groups[i].payloads[j].GetPath()
                        tempPayload = RigidPrimView(
                            f"{tempPayload_path}",
                            reset_xform_properties=False,
                            track_contact_forces=False
                        )
                        current_payload_pos, current_payload_rot = self.get_env_poses(tempPayload.get_world_poses())

                        _payload = DisconnectedPayload(
                            payload.type,
                            payload.detail().target_pos,
                            current_payload_pos.squeeze(axis=0),
                            current_payload_rot.squeeze(axis=0)
                        )
                        payloads.append(_payload)
                if group_snapshot.stage == Stage.TRANSPORT:
                    target_payload_idx = group_snapshot.target_payload_idx + 1 if group_snapshot.target_payload_idx < self.num_payloads_per_group - 1 else group_snapshot.target_payload_idx
            else:
                stage = group_snapshot.stage
                target_payload_idx = group_snapshot.target_payload_idx
                payloads = []
                for j, payload in enumerate(group_snapshot.payloads):
                    if isinstance(payload, ConnectedPayload):
                        pos, rot = self.groups[i].transport.get_world_poses(True)
                        vel = self.groups[i].transport.get_velocities(True)
                        joint_pos = self.groups[i].transport.get_joint_positions(True)
                        joint_vel = self.groups[i].transport.get_joint_velocities(True)

                        payloads.append(ConnectedPayload(
                            payload.type,
                            payload.detail().target_pos,
                            pos.squeeze(axis=0),
                            rot.squeeze(axis=0),
                            vel.squeeze(axis=0),
                            joint_pos.squeeze(axis=0),
                            joint_vel.squeeze(axis=0)
                        ))
                    else:
                        tempPayload_path = self.groups[i].payloads[j].GetPath()
                        tempPayload = RigidPrimView(
                            f"{tempPayload_path}",
                            reset_xform_properties=False,
                            track_contact_forces=False
                        )
                        current_payload_pos, current_payload_rot = self.get_env_poses(tempPayload.get_world_poses())

                        if j < group_snapshot.target_payload_idx:
                            current_payload_rot[0][0] = 0
                            current_payload_rot[0][1] = 0
                            current_payload_rot[0][2] = 0

                        _payload = DisconnectedPayload(
                            payload.type,
                            payload.detail().target_pos,
                            current_payload_pos.squeeze(axis=0),
                            current_payload_rot.squeeze(axis=0)
                        )
                        payloads.append(_payload)

            group_snapshot = GroupSnapshot(
                drone_pos,
                drone_rot,
                drone_vel,
                target_payload_idx,
                stage,
                count,
                payloads
            )

            group_snapshots.append(group_snapshot)

        return StateSnapshot(group_snapshots, self.done_payloads)

    def make_group_offset(self):
        group_interval = 10
        group_offset = torch.zeros(self.num_groups, 3, device=self.device)
        group_offset[:, 0] = torch.arange(start=0, end=-(group_interval * self.num_groups), step=-group_interval,
                                          device=self.device)

        return group_offset

    def make_payload_offset(self):
        payload_offset = []
        for i in range(self.num_payloads_per_group):
            payload_position = [3 - i * 2, 0, 0]
            payload_offset.append(payload_position)
        return torch.FloatTensor(payload_offset).to(device=self.device)

    def make_initial_state(self):
        payload_pos_dist = D.Uniform(
            torch.tensor([-0.5, -0.5, 0], device=self.device),
            torch.tensor([0.5, 0.5, 0], device=self.device)
        )
        
        payload_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )

        groups = []
        payload_type = torch.zeros(self.num_groups, self.num_payloads_per_group)
        # for i in range(self.num_groups):
        #     payload_type[i] = torch.randperm(Payload.__len__())[:self.num_payloads_per_group]
        payload_type[0] = torch.tensor([0,2,4])
        payload_type[1] = torch.tensor([4,0,2])

        for i in range(self.num_groups):
            drone_pos = self.formation + self.group_offset[i]
            drone_rot = torch.zeros((self.num_drones_per_group, 4), device=self.device)
            drone_rot[:, 0] = 1
            drone_vel = torch.zeros((self.num_drones_per_group, 6), device=self.device)
            target_payload_idx = 0
            stage = Stage.FORMATION
            count = 0
            payloads = []
            for j in range(self.num_payloads_per_group):
                payload = Payload.get(payload_type[i][j])
                payload_pos = payload_pos_dist.sample() + self.group_offset[i] + self.payload_offset[j]
                if payload.name == "D1" or payload.name == "CC1":
                    payload_pos[2] = 0.05
                elif payload.name == "D1_s" or payload.name == "CC2":
                    payload_pos[2] = 0.0125
                elif payload.name == "A1" or payload.name == "B1" or payload.name == "CA1" or payload.name == "CB1":
                    payload_pos[2] = 0.1
                elif payload.name == "A2" or payload.name == "B2" or payload.name == "CA2" or payload.name == "CB2":
                    payload_pos[2] = 0.0625
                payload_rot = euler_to_quaternion(payload_rpy_dist.sample())
                payloads.append(DisconnectedPayload(payload, payload.value.target_pos, payload_pos, payload_rot))

            groups.append(
                GroupSnapshot(drone_pos, drone_rot, drone_vel, target_payload_idx, stage, count, payloads)
            )
        done_payloads = {payload.name: 0 for payload in Payload}
        return StateSnapshot(groups, done_payloads)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        scene_utils.design_scene()

        if self.enable_background:

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A1.usd"
            prim_utils.create_prim("/World/envs/Rack1", usd_path=asset_path, translation=(-9, -9, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A2.usd"
            prim_utils.create_prim("/World/envs/Rack2", usd_path=asset_path, translation=(-4, -9, 0), scale=(0.01, 0.01, 0.01),orientation = (0.7071068, 0, 0, 0.7071068))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A3.usd"
            prim_utils.create_prim("/World/envs/Rack3", usd_path=asset_path, translation=(1, -9, 0), scale=(0.01, 0.01, 0.01),orientation = (0.7071068, 0, 0, 0.7071068))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A4.usd"
            prim_utils.create_prim("/World/envs/Rack4", usd_path=asset_path, translation=(-9, -3, 0), scale=(0.01, 0.01, 0.01))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A5.usd"
            prim_utils.create_prim("/World/envs/Rack5", usd_path=asset_path, translation=(-9, 3, 0), scale=(0.01, 0.01, 0.01))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A6.usd"
            prim_utils.create_prim("/World/envs/Rack6", usd_path=asset_path, translation=(-9, 9, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/warehouse_test.usd"
            prim_utils.create_prim("/World/envs/Warehouse", usd_path=asset_path, translation=(0, 10, 0.01), scale=(0.02, 0.02, 0.02))

            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A6.usd"
            prim_utils.create_prim("/World/envs/Rack7", usd_path=asset_path, translation=(-9, 12, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A7.usd"
            prim_utils.create_prim("/World/envs/Rack8", usd_path=asset_path, translation=(-4, 12, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A8.usd"
            prim_utils.create_prim("/World/envs/Rack9", usd_path=asset_path, translation=(1, 12, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLarge_A9.usd"
            prim_utils.create_prim("/World/envs/Rack10", usd_path=asset_path, translation=(6, 12, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A1.usd"
            prim_utils.create_prim("/World/envs/Rack11", usd_path=asset_path, translation=(11, 12, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))


            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack12", usd_path=asset_path, translation=(-15, 10, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack13", usd_path=asset_path, translation=(-15, 15, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack14", usd_path=asset_path, translation=(-15, 5, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack15", usd_path=asset_path, translation=(-15, 0, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack16", usd_path=asset_path, translation=(-15, -5, 0), scale=(0.01, 0.01, 0.01))


            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack17", usd_path=asset_path, translation=(-15, -15, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack18", usd_path=asset_path, translation=(-10, -15, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack19", usd_path=asset_path, translation=(-5, -15, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A8.usd"
            prim_utils.create_prim("/World/envs/Rack20", usd_path=asset_path, translation=(0, -15, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack21", usd_path=asset_path, translation=(5, -15, 0), scale=(0.01, 0.01, 0.01), orientation = (0.7071068, 0, 0, 0.7071068))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack22", usd_path=asset_path, translation=(11, 0, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack23", usd_path=asset_path, translation=(11, -5, 0), scale=(0.01, 0.01, 0.01))
            asset_path = ASSET_PATH + "/industry_usd/Warehouse/Racks/RackLong_A9.usd"
            prim_utils.create_prim("/World/envs/Rack24", usd_path=asset_path, translation=(11, 5, 0), scale=(0.01, 0.01, 0.01))

        for i, group_snapshot in enumerate(self.initial_state.group_snapshots):
            drones = drone_model(cfg=cfg)
            transport = None
            payloads = []
            group_prim_path = f"/World/envs/env_0/Group_{i}"
            # spawn drones
            if group_snapshot.stage == Stage.PRE_TRANSPORT or group_snapshot.stage == Stage.TRANSPORT:
                group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones_per_group)
                payload = group_snapshot.payloads[group_snapshot.target_payload_idx]
                payload_position = payload.payload_pos.clone().detach()
                # drone_translation = drone_poses - payload_position
                # drone_transition[:,2] = 0
                transport = TransportationGroup(drone=drones, cfg=group_cfg)
                transport.spawn(
                    translations=payload_position,
                    prim_paths=[group_prim_path],
                    enable_collision=True,
                    payload_usd=payload.detail().usd_path,
                    payload_scale=payload.detail().scale,
                    name = payload.detail().name
                )

                DynamicCuboid(
                    "/World/envs/env_0/payloadTargetVis{}".format(i),
                    position=group_snapshot.target_payload().target_pos,
                    scale=group_snapshot.target_payload().detail().shadow_scale,
                    orientation=(0.7071068,0,0,0),
                    color=torch.tensor([0.8, 0.1, 0.1]),
                    size=2.01,
                )
                kit_utils.set_collision_properties(
                    "/World/envs/env_0/payloadTargetVis{}".format(i),
                    collision_enabled=False
                )
                kit_utils.set_rigid_body_properties(
                    "/World/envs/env_0/payloadTargetVis{}".format(i),
                    disable_gravity=True
                )
            else:
                prim_utils.create_prim(group_prim_path)  # xform
                drone_prim_paths = [f"{group_prim_path}/{drones.name.lower()}_{j}" for j in
                                    range(self.num_drones_per_group)]
                drones.spawn(translations=group_snapshot.drone_pos, prim_paths=drone_prim_paths)

            # spawn payload
            for j, payload in enumerate(group_snapshot.payloads):
                if isinstance(payload, DisconnectedPayload):
                    temp_payload = self.create_payload(payload.payload_pos,
                                                       f"{group_prim_path}/payload_{j}",
                                                       payload.detail().usd_path,
                                                       payload.detail().scale,
                                                       payload.payload_rot)  #
                    payloads.append(temp_payload)
                else:
                    payloads.append(None)

            self.groups.append(Group(drones, payloads, transport))

    def _reset_idx(self, env_ids: torch.Tensor):
        for i, group in enumerate(self.groups):
            group_snapshot = self.initial_state.group_snapshots[i]
            pos = group_snapshot.drone_pos.clone().detach()
            rot = group_snapshot.drone_rot.clone().detach()
            vel = group_snapshot.drone_vel.clone().detach()

            if group.transport is not None: #########여기서 드론 rot과 vel을 설정해야한다는 듯?
                payload = group_snapshot.target_payload()
                group.transport._reset_idx(env_ids)
                group.transport.set_world_poses(payload.payload_pos, payload.payload_rot, env_ids)
                group.transport.set_velocities(payload.payload_vel, env_ids)
                group.transport.set_joint_positions(payload.joint_pos, env_ids)
                group.transport.set_joint_velocities(payload.joint_vel, env_ids)
                group.transport.payload_view.set_masses(torch.tensor([1.36], device=self.device), env_ids)
                group.transport.drone.set_world_poses(pos, rot, env_ids)
                group.transport.drone.set_velocities(vel, env_ids)
            else:
                group.drones._reset_idx(env_ids)
                group.drones.set_world_poses(pos, rot, env_ids)
                group.drones.set_velocities(vel, env_ids)

    def _set_specs(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        drone = drone_model(cfg=cfg)

        drone_state_dim = drone.state_spec.shape[0]

        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone_num - 1, 13 + 1)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone_num, drone_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone_num),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([drone.action_spec] * self.drone_num, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone_num, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone_num,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        for i, start in enumerate(torch.arange(0, self.drone_num, self.num_drones_per_group)):
            end = start + self.num_drones_per_group
            action = actions[:, start:end, :]
            self.groups[i].drones.apply_action(action)

    def _compute_state_and_obs(self):
        return TensorDict({
            "agents": {
                "observation": TensorDict({}, [self.num_envs, self.drone_num]),
                "observation_central": TensorDict({}, self.batch_size),
            }
        }, self.batch_size)

    def get_formation_state(self, group_idx=0):
        group_snapshot = self.initial_state.group_snapshots[group_idx]
        root_states = self.groups[group_idx].drones.get_state()
        pos = self.groups[group_idx].drones.pos
        payload = group_snapshot.payloads[group_snapshot.target_payload_idx]
        target_pos = payload.payload_pos.clone().detach()
        target_pos[2] += 1
        root_states[..., :3] = target_pos - pos

        self.envOneHot = torch.zeros(self.num_envs, len(Payload), device=self.device)
        index = list(Payload).index(Payload[group_snapshot.target_payload().detail().name])

        payloadOneHot = np.zeros((1,len(Payload)))
        payloadOneHot[np.arange(len(payloadOneHot)), index] = 1
        self.envOneHot[0] = torch.FloatTensor(payloadOneHot).to(self.device)

        obs_self = [root_states]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.num_drones_per_group, self.time_encoding_dim))
        obs_self.append(self.envOneHot.expand(self.drone_num // len(self.groups), self.num_envs, len(Payload)).transpose(0, 1))
        obs_self = torch.cat(obs_self, dim=-1)



        relative_pos = torch.vmap(cpos)(pos, pos)
        drone_pdist = torch.vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
        relative_pos = torch.vmap(off_diag)(relative_pos)

        # print(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            drone_pdist,
            torch.vmap(others)(root_states[..., 3:13])
        ], dim=-1)

        obs = TensorDict({
            "obs_self": obs_self.unsqueeze(2),
            "obs_others": obs_others,
            "pos": self.groups[group_idx].drones.pos,
        }, [self.num_envs, self.num_drones_per_group])

        state = TensorDict({"drones":root_states}, self.batch_size)

        return TensorDict({
            "agents": {
                "observation": obs,
                "observation_central": state,
            }
        }, self.batch_size)

    def get_transport_state(self, group_idx=0):
        group_snapshot = self.initial_state.group_snapshots[group_idx]
        drone_states = self.groups[group_idx].drones.get_state()
        payload = self.groups[group_idx].transport.payload_view
        payload_vels = payload.get_velocities()
        drone_pos = drone_states[..., :3]

        payload_pos, payload_rot = self.get_env_poses(payload.get_world_poses())
        payload_heading: torch.Tensor = quat_axis(payload_rot, axis=0)
        payload_up: torch.Tensor = quat_axis(payload_rot, axis=2)

        drone_rpos = torch.vmap(cpos)(drone_pos, drone_pos)
        drone_rpos = torch.vmap(off_diag)(drone_rpos)
        drone_pdist = torch.norm(drone_rpos, dim=-1, keepdim=True)
        payload_drone_rpos = payload_pos.unsqueeze(1) - drone_pos

        payload_target_pos = torch.tensor(group_snapshot.payloads[group_snapshot.target_payload_idx].target_pos, device=self.device)
        payload_target_heading = torch.zeros(1, 3, device=self.device)

        target_payload_rpose = torch.cat([
            payload_target_pos - payload_pos,
            payload_target_heading - payload_heading
        ], dim=-1)

        payload_state = [
            target_payload_rpose,
            payload_rot,  # 4
            payload_vels,  # 6
            payload_heading,  # 3
            payload_up,  # 3
        ]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            payload_state.append(t.expand(-1, self.time_encoding_dim))
        payload_state = torch.cat(payload_state, dim=-1).unsqueeze(1)

        obs = TensorDict({}, [self.num_envs, self.num_drones_per_group])
        identity = torch.eye(self.num_drones_per_group, device=self.device).expand(self.num_envs, -1, -1)
        obs["obs_self"] = torch.cat(
            [-payload_drone_rpos, drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)  # [..., 1, state_dim]
        obs["obs_others"] = torch.cat(
            [drone_rpos, drone_pdist, torch.vmap(others)(drone_states[..., 3:13])], dim=-1
        )  # [..., n-1, state_dim + 1]
        obs["obs_payload"] = payload_state.expand(-1, self.num_drones_per_group, -1).unsqueeze(2)  # [..., 1, 22]

        state = TensorDict({}, self.num_envs)
        state["payload"] = payload_state  # [..., 1, 22]
        state["drones"] = obs["obs_self"].squeeze(2)  # [..., n, state_dim]

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
            }
        }, self.num_envs)

    def _compute_reward_and_done(self):
        done = False
        for i, group_snapshot in enumerate(self.initial_state.group_snapshots):
            if group_snapshot.stage == Stage.FORMATION:
                payload = group_snapshot.payloads[group_snapshot.target_payload_idx]
                pos = self.groups[i].drones.pos
                target_pos = payload.payload_pos.clone().detach()
                target_pos[2] += 1
                distance = torch.norm(pos.mean(-2, keepdim=True) - target_pos, dim=-1)
                terminated = (distance < 0.2)
            elif group_snapshot.stage == Stage.POST_FORMATION:
                self.count[i] += 1
                terminated = (self.count[i] > 70)
            elif group_snapshot.stage == Stage.PRE_TRANSPORT:
                self.count[i] += 1
                terminated = (self.count[i] > 150)
            elif group_snapshot.stage == Stage.TRANSPORT:
                payload = self.groups[i].transport.payload_view
                payload_target_heading = torch.zeros(1, 3, device=self.device)

                payload_pos, payload_rot = self.get_env_poses(payload.get_world_poses())
                payload_heading: torch.Tensor = quat_axis(payload_rot, axis=0)

                payload = group_snapshot.payloads[group_snapshot.target_payload_idx]
                payload_target_pos = torch.tensor(payload.target_pos, device=self.device)

                target_payload_rpose = torch.cat([
                    payload_target_pos - payload_pos,
                    payload_target_heading - payload_heading], dim=-1)

                p_distance = torch.norm(target_payload_rpose, dim=-1, keepdim=True)

                if p_distance < 1.002:
                    self.count[i] += 1
                terminated = (self.count[i] > 5)
            else:
                raise NotImplementedError

            truncated = (self.progress_buf >= self.max_episode_length)

            if terminated | truncated:
                done = True
                self.done_group = i
                self.count[i] = 0

        return TensorDict(
            {
                "agents": {
                    "reward":  torch.FloatTensor([[[0]]]).to(device=self.device).expand(-1, self.drone_num, 1)
                },
                "done": torch.tensor([done],device=self.device),
                "terminated": torch.tensor([False],device=self.device),
                "truncated": torch.tensor([False],device=self.device),
            },
            self.batch_size,
        )

    def create_payload(self, pos, prim_path, usd_path, scale, rot):
        payload = prim_utils.create_prim(
            prim_path=prim_path,
            usd_path=usd_path,
            position=pos,
            orientation=rot,
            scale=scale,
        )

        script_utils.setRigidBody(payload, "convexHull", False)
        UsdPhysics.MassAPI.Apply(payload)
        payload.GetAttribute("physics:mass").Set(1.36)
        payload.GetAttribute("physics:rigidBodyEnabled").Set(True)

        restitution_attr = payload.GetAttribute("physics:restitution")
        if not restitution_attr:
            restitution_attr = payload.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float, True)
        restitution_attr.Set(0.0)

        kit_utils.set_rigid_body_properties(
            payload.GetPath(),
            angular_damping=0.1,
            linear_damping=0.1
        )
        return payload
