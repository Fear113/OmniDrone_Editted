from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class Payload:
    target_pos: torch.Tensor
    target_rot: torch.Tensor
@dataclass
class ConnectedPayload(Payload):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

@dataclass
class DisconnectedPayload(Payload):
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor

@dataclass
class Group:
    drone_pos: torch.Tensor
    drone_rot: torch.Tensor
    drone_vel: torch.Tensor
    target_payload_idx: Optional[int]
    is_transporting: bool
    payloads: list[ConnectedPayload | DisconnectedPayload]

@dataclass
class StateSnapshot:
    groups: list[Group]
