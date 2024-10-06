from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

class Stage(Enum):
    FORMATION=0
    POST_FORMATION=1
    PRE_TRANSPORT=2
    TRANSPORT=3

    def next(self):
        return Stage((self.value + 1) % Stage.__len__())

@dataclass
class Payload:
    usd_path: str
    scale: tuple[float, float, float]
    target_pos: torch.Tensor
    target_rot: torch.Tensor
    name: str
@dataclass
class ConnectedPayload(Payload):
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor
    payload_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

@dataclass
class DisconnectedPayload(Payload):
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor

@dataclass
class GroupSnapshot:
    drone_pos: torch.Tensor
    drone_rot: torch.Tensor
    drone_vel: torch.Tensor
    target_payload_idx: Optional[int]
    stage: Stage
    count: int
    payloads: list[ConnectedPayload | DisconnectedPayload]

    def target_payload(self):
        return self.payloads[self.target_payload_idx]

@dataclass
class StateSnapshot:
    group_snapshots: list[GroupSnapshot]
