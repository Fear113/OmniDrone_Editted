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


import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
import numpy as np
from omni_drones.utils.payload import Payload
import omni.physx.scripts.utils as script_utils
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import DynamicSphere

targetposition = [0,0,0]

REGULAR_HEXAGON = [
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [0, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]
REGULAR_SQUARE =[
    [1,1,1],
    [1,-1,1],
    [-1,-1,1],
    [-1,1,1],
]

DOUBLE = [
    [1.5, 1, 1],
    [1.5, -1, 1],
    [-1.5, -1, 1],
    [-1.5, 1, 1]
]

FLAT_S = [
    [1.2, 0.7, 0.5],
    [1.2, -0.7, 0.5],
    [-1.2, -0.7, 0.5],
    [-1.2, 0.7, 0.5]
]

FLAT_L = [
    [1.5, 1.0, 0.5],
    [1.5, -1.0, 0.5],
    [-1.5, -1.0, 0.5],
    [-1.5, 1.0, 0.5]
]

FORMATIONS = {
    # "squre": REGULAR_SQUARE,
    "squre": REGULAR_SQUARE,
    "squre_2": DOUBLE,
    "flat_s": FLAT_S,
    "flat_l": FLAT_L,
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
}

FORMATION_A1 = [ # 정육면체 
    [0.48,0.48,0.9],
    [0.48,-0.48,0.9],
    [-0.48,0.48,0.9],
    [-0.48,-0.48,0.9]
]

FORMATION_A2 = [ # 정육면체, 좀 더 작음. 지나치게 딱 달라붙어있어 학습에 어려움이 있지 않을까 걱정을 하고있긴 함. 
    [0.25,0.25,0.5],
    [0.25,-0.25,0.5],
    [-0.25,0.25,0.5],
    [-0.25,-0.25,0.5]
]
FORMATION_B1 = [ # 직육면체, 정육면체 2개를 이어붙인 구조
    [0.48, 0.96, 0.9],
    [0.48, -0.96, 0.9],
    [-0.48, 0.96, 0.9],
    [-0.48, -0.96, 0.9]
]
FORMATION_B2 = [ # 직육면체, 정육면체 2개를 이어붙인 구조. 좀 더 작음
    [0.25, 0.50, 0.5],
    [0.25, -0.50, 0.5],
    [-0.25, 0.50, 0.5],
    [-0.25, -0.50, 0.5]
]
FORMATION_D1 = [ # 직육면체, 정육면체 6개를 이어붙인 구조. 
    [0.55, 0.85, 0.6],
    [0.55, -0.85, 0.6],
    [-0.55, 0.85, 0.6],
    [-0.55, -0.85, 0.6]
]
FORMATION_D1_S = [ #직육면체, 정육면체 6개를 이어붙인 구조. 좀 더 작음. 
    [0.2, 0.4, 0.1],
    [0.2, -0.4, 0.1],
    [-0.2, 0.4, 0.1],
    [-0.2, -0.4, 0.1]
]
### Belows are old formations for cardbox. Please remain this formations. 
# FORMATION_CA1 = [ #직육면체 
#     [0.45, 0.4, 0.75],
#     [0.45, -0.4, 0.75],
#     [-0.45, 0.4, 0.75],
#     [-0.45, -0.4, 0.75]
# ]

# FORMATION_CA2 = [ #직육면체 
#     [0.4, 0.35, 0.6],
#     [0.4, -0.35, 0.6],
#     [-0.4, 0.35, 0.6],
#     [-0.4, -0.35, 0.6]
# ]

# FORMATION_CB1 = [ #직육면체 (세로)
#     [0.35, 0.7, 0.85],
#     [0.35, -0.7, 0.85],
#     [-0.35, 0.7, 0.85],
#     [-0.35, -0.7, 0.85]
# ]


# FORMATION_CB2 = [ #직육면체 (세로) 너무 딱 붙어있어야 하는 구조가 나와서 살짝 힘들듯 하다. 그래서 모양이 안맞더라도 좀 떨어지게 해놨음. 원래는 0.3이 아닌, 0.25여야함. 
#     [0.3, 0.6, 0.7],
#     [0.3, -0.6, 0.7],
#     [-0.3, 0.6, 0.7],
#     [-0.3, -0.6, 0.7]
# ]


# FORMATION_CC1 = [ #직육면체 (세로, 낮음)
#     [0.6, 0.8, 0.7],
#     [0.6, -0.8, 0.7],
#     [-0.6, 0.8, 0.7],
#     [-0.6, -0.8, 0.7]
# ]

# FORMATION_CC2 = [ #직육면체 (세로, 낮음)
#     [0.45, 0.6, 0.5],
#     [0.45, -0.6, 0.5],
#     [-0.45, 0.6, 0.5],
#     [-0.45, -0.6, 0.5]
# ]

MYFORMATION = np.array([
    FORMATION_A1, FORMATION_A2, FORMATION_B1, FORMATION_B2, FORMATION_D1, FORMATION_D1_S
])

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class Formation(IsaacEnv):
    """
    This is a formation control task. The goal is to control the drone to form a
    regular polygon formation. The reward is the negative of the formation cost.

    ## Observation
    - `obs_self`: the relative position, velocity, and orientation of the drone
    - `obs_others`: the relative position, velocity, and orientation of other drones

    ## Reward
    
    - `formation`: the negative of the formation cost.
    - `pos`: the negative of the distance to the target position.
    - `heading`: the negative of the heading error.

    ## Episode End

    The episode terminates when any of the following conditions are met:
    - The drone crashes.
    - The minimum distance between any two drones is less than a threshold.

    or is truncated when it reaches the maximum length.

    ## Config 

    """
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        # initial state distribution
        self.cells = (
            make_cells([-25, -25, 0.25], [25, 25, 3], [5.0, 5.0, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )

        self.middle_pos_dist = D.Uniform(
            torch.tensor([-20, -20, 1.0], device=self.device),
            torch.tensor([20, 20, 2.5], device=self.device)
        )
        self.spead_pos_dist = D.Uniform(
            torch.tensor([-5, -5, -0.5], device=self.device),
            torch.tensor([5, 5, 0.5], device=self.device)
        )

        # to default
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_pos = self.target_pos.expand(self.num_envs, 1, 3)
        self.target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading[..., 0] = -1

        self.alpha = 0.8

        # self.last_cost_l = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device=self.device)
        self.envOneHot = torch.zeros(self.num_envs, len(Payload), device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()
        target_pos_numpy = np.random.uniform(low=-2, high=2, size=3)
        self.target_pos = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        
        

        # formation = self.cfg.task.formation
        # if isinstance(formation, str):
        #     self.formation = torch.as_tensor(
        #         FORMATIONS[formation], device=self.device
        #     ).float()
        # elif isinstance(formation, list):
        #     self.formation = torch.as_tensor(
        #         self.cfg.task.formation, device=self.device
        #     )
        # else:
        #     raise ValueError(f"Invalid target formation {formation}")
        
        
        # self.formation_L = laplacian(self.formation)
        init_position = torch.as_tensor([[1.0,1.0,1.0],[1.0,-1.0,1.0],[-1.0,1.0,1.0],[-1.0,-1.0,1.0]],device=self.device)
        self.drone.spawn(init_position) 

        self.formation = torch.FloatTensor(np.zeros((self.num_envs,self.drone.n, 3))).to(self.device)
        self.formation = self.formation + self.target_pos 

        
        
        
        
        
        ######## Please remain below comments. It will use again after adding payload types.
        # index = 5
        # USD_PATH = list(Payload)[index].value.usd_path
        # SCALE = list(Payload)[index].value.scale
        # payload_rpy_dist = D.Uniform(
        #     torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
        #     torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        # )
        # ROT = euler_to_quaternion(payload_rpy_dist.sample())
        # create_payload([0,0,0.5],
        #             "/World/envs/env_0/payload_0",
        #             USD_PATH,
        #             SCALE,
        #             ROT) 

        # DynamicSphere(
        # "/World/envs/env_0/goal1",
        # translation=torch.tensor(MYFORMATION[index][0]),
        # color=torch.tensor([1.0, 0.2, 0.2]),
        # radius=0.3,
        # mass=0.1,
        # )

        # DynamicSphere(
        # "/World/envs/env_0/goal2",
        # translation=torch.tensor(MYFORMATION[index][1]),
        # color=torch.tensor([1.0, 0.2, 0.2]),
        # radius=0.3,
        # mass=0.1,
        # )

        # DynamicSphere(
        # "/World/envs/env_0/goal3",
        # translation=torch.tensor(MYFORMATION[index][2]),
        # color=torch.tensor([1.0, 0.2, 0.2]),
        # radius=0.3,
        # mass=0.1,
        # )

        # DynamicSphere(
        # "/World/envs/env_0/goal4",
        # translation=torch.tensor(MYFORMATION[index][3]),
        # color=torch.tensor([1.0, 0.2, 0.2]),
        # radius=0.3,
        # mass=0.1,
        # )


        # kit_utils.set_collision_properties(
        #     "/World/envs/env_0/goal1",
        #     collision_enabled=True
        # )
        # kit_utils.set_rigid_body_properties(
        #     "/World/envs/env_0/goal1",
        #     disable_gravity=True
        # )

        # kit_utils.set_collision_properties(
        #     "/World/envs/env_0/goal2",
        #     collision_enabled=True
        # )
        # kit_utils.set_rigid_body_properties(
        #     "/World/envs/env_0/goal2",
        #     disable_gravity=True
        # )

        # kit_utils.set_collision_properties(
        #     "/World/envs/env_0/goal3",
        #     collision_enabled=True
        # )
        # kit_utils.set_rigid_body_properties(
        #     "/World/envs/env_0/goal3",
        #     disable_gravity=True
        # )

        # kit_utils.set_collision_properties(
        #     "/World/envs/env_0/goal4",
        #     collision_enabled=True
        # )
        # kit_utils.set_rigid_body_properties(
        #     "/World/envs/env_0/goal4",
        #     disable_gravity=True
        # )
        DynamicSphere(
        "/World/envs/env_0/goal",
        translation=torch.tensor([0., 0., 1.5]),
        color=torch.tensor([1.0, 0.2, 0.2]),
        radius=0.3,
        mass=0.1,
        )
        kit_utils.set_collision_properties(
            "/World/envs/env_0/goal",
            collision_enabled=True
        )
        kit_utils.set_rigid_body_properties(
            "/World/envs/env_0/goal",
            disable_gravity=True
        )
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim
        payload_type_dim = len(Payload)
        obs_self_dim += payload_type_dim
        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)),
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
            action_key=("agents","action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )
         # additional infos & buffers
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            # "cost_laplacian": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            "cost_hausdorff": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        payloadIndex = np.random.randint(len(Payload), size=len(env_ids))
        # payloadIndex = np.random.randint(1,size=len(env_ids))
        # payloadIndex = 6
        payloadOneHot = np.zeros((len(env_ids),len(Payload)))
        payloadOneHot[np.arange(len(payloadOneHot)), payloadIndex] = 1
        self.envOneHot[env_ids] = torch.FloatTensor(payloadOneHot).to(self.device)

        ####### target payload
        self.formation[env_ids] = torch.FloatTensor(MYFORMATION[payloadIndex]).to(self.device) ################# need to change formation format as tensor with index format

        self.formation[env_ids] += self.target_pos[env_ids]
        
        if (self.formation[:,:,2] > 2).any():
            print("hi")
        middle_point = self.middle_pos_dist.sample(env_ids.shape)
        pos = middle_point.repeat(4,1,1).transpose(0,1) + self.spead_pos_dist.sample(torch.tensor([env_ids.shape[0]*4])).reshape(env_ids.shape[0],4,3) + self.envs_positions[env_ids].unsqueeze(1)
        # pos = torch.vmap(sample_from_grid, randomness="different")(
        #     self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        # ) + self.envs_positions[env_ids].unsqueeze(1)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)

        temp = torch.vmap(cost_formation_hausdorff)(
        pos, self.formation[env_ids])
        self.last_cost_h[env_ids] = temp.unsqueeze(dim=1)
        # self.last_cost_l[env_ids] = vmap(cost_formation_laplacian)(
        #     pos, desired_p=self.formation
        # )
        com_pos = (pos - self.envs_positions[env_ids].unsqueeze(1)).mean(1, keepdim=True)
        self.last_cost_pos[env_ids] = torch.square(
            com_pos - self.target_pos[env_ids]
        ).sum(2)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_states = self.drone.get_state()
        pos = self.drone.pos
        self.root_states[..., :3] = self.target_pos - pos

        obs_self = [self.root_states]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        obs_self.append(self.envOneHot.expand(self.drone.n, self.num_envs, len(Payload)).transpose(0, 1))
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
        }, [self.num_envs, self.drone.n])

        state = TensorDict({"drones": self.root_states}, self.batch_size)

        return TensorDict({
            "agents": {
                "observation": obs, 
                "observation_central": state,
            },
            "stats": self.stats
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # cost_l = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L)
        pos = self.drone.pos

        cost_h = torch.vmap(cost_formation_hausdorff)(pos, self.formation).unsqueeze(dim=1)
        
        distance = torch.norm(pos.mean(-2, keepdim=True) - self.target_pos, dim=-1)

        reward_formation =  1 / (1 + torch.square(cost_h * 1.6)) 
        # reward_pos = 1 / (1 + cost_pos)

        # reward_formation = torch.exp(- cost_h * 1.6)
        reward_pos = torch.exp(- distance) *2
        reward_heading = self.drone.heading[..., 0].mean(-1, True)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward = (
            reward_separation * (
                reward_formation 
                + reward_formation * (reward_pos + reward_heading)
                + 0.6 * reward_pos * reward_formation
            )
        )

        # self.last_cost_l[:] = cost_l
        self.last_cost_h[:] = cost_h
        self.last_cost_pos[:] = torch.square(distance)

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)

        terminated = crash | (separation<0.1)

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        # self.stats["cost_laplacian"] -= cost_l
        self.stats["cost_hausdorff"].lerp_(cost_h, (1-self.alpha))
        self.stats["pos_error"].lerp_(distance, (1-self.alpha))

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


def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    symmetric normalized laplacian

    p: (n, dim)
    """
    assert p.dim() == 2
    A = torch.cdist(p, p)
    D = torch.sum(A, dim=-1)
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = D - A
    return L

def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    if p.dim()==1:
        p = p.reshape(4,3)
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost
def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d



def create_payload(pos, prim_path, usd_path, scale, rot):
    

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

    kit_utils.set_rigid_body_properties(
        payload.GetPath(),
        angular_damping=0.1,
        linear_damping=0.1
    )
    return payload