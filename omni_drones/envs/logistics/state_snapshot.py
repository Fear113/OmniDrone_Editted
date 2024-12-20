from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from omni_drones.utils.payload import Payload


class Stage(Enum):
    FORMATION=0
    POST_FORMATION=1
    PRE_TRANSPORT=2
    TRANSPORT=3

    def next(self):
        return Stage((self.value + 1) % Stage.__len__())

@dataclass
class _Payload:
    type: Payload
    target_pos: tuple[float, float, float]

    def detail(self):
        return self.type.value

@dataclass
class ConnectedPayload(_Payload):
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor
    payload_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

@dataclass
class DisconnectedPayload(_Payload):
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
    n_done: int

    def target_payload(self):
        return self.payloads[self.target_payload_idx]

@dataclass
class StateSnapshot:
    group_snapshots: list[GroupSnapshot]
    stacked_payload: list[DisconnectedPayload]
    done_payloads: dict[str, int]