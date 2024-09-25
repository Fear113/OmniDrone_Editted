from dataclasses import dataclass
from typing import Optional

import torch

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
    is_transporting: bool
    count: int
    payloads: list[ConnectedPayload | DisconnectedPayload]

    def target_payload(self):
        return self.payloads[self.target_payload_idx]

@dataclass
class StateSnapshot:
    group_snapshots: list[GroupSnapshot]
