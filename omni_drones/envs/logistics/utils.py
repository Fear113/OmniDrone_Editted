from dataclasses import dataclass

import torch

@dataclass
class Payload:
    target_pos: torch.Tensor
    target_rot: torch.Tensor

class ConnectedPayload(Payload):
    pass

@dataclass
class DisconnectedPayload(Payload):
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor

@dataclass
class Group:
    drone_pos: torch.Tensor
    drone_rot: torch.Tensor
    drone_vel: torch.Tensor
    target_payload_idx: int
    is_transporting: bool
    payloads: list[ConnectedPayload | DisconnectedPayload]

@dataclass
class StateSnapshot:
    groups: list[Group]
