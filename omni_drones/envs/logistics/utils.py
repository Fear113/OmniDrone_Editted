from dataclasses import dataclass

import torch

class ConnectedPayload:
    pass

@dataclass
class DisconnectedPayload:
    payload_pos: torch.Tensor
    payload_rot: torch.Tensor

@dataclass
class Group:
    drone_pos: torch.Tensor
    drone_rot: torch.Tensor
    drone_vel: torch.Tensor
    is_transporting: bool
    payloads: list[ConnectedPayload | DisconnectedPayload]

@dataclass
class InitialState:
    groups: list[Group]
