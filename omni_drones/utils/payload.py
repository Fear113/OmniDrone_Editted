from enum import Enum

from attr import dataclass

import os.path as osp

from omni_drones.robots import ASSET_PATH

dir = ASSET_PATH + "/industry_usd/RackLarge/Containers/Wooden"

@dataclass
class PayloadInfo:
    usd_path: str
    scale: tuple[float, float, float]
    target_pos: tuple[float, float, float]
    name: str

class PayloadList(Enum):
    A1 = PayloadInfo(f"{dir}/WoodenCrate_A1.usd", (0.008, 0.008, 0.008), (3., 4, 0), 'A1')
    A2 = PayloadInfo(f"{dir}/WoodenCrate_A2.usd", (0.0065, 0.0065, 0.0065), (2., 4, 0), 'A2')
    B1 = PayloadInfo(f"{dir}/WoodenCrate_B1.usd", (0.008, 0.008, 0.008), (1., 4, 0), 'B1')
    B2 = PayloadInfo(f"{dir}/WoodenCrate_B2.usd", (0.0065, 0.0065, 0.0065), (0., 4, 0), 'B2')
    D1 = PayloadInfo(f"{dir}/WoodenCrate_D1.usd", (0.006, 0.006, 0.006), (-1., 4, 0), 'D1')
    D1_s = PayloadInfo(f"{dir}/WoodenCrate_D1.usd", (0.0045, 0.0045, 0.0045), (-2., 4, 0), 'D1_s')