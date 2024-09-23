from enum import Enum

from attr import dataclass

import os.path as osp

from omni_drones.robots import ASSET_PATH

dir = ASSET_PATH + "/industry_usd/RackLarge/Containers/Wooden"

@dataclass
class Spawn:
    usd_path: str
    scale: tuple[float, float, float]

class Payload(Enum):
    A1 = Spawn(f"{dir}/WoodenCrate_A1.usd", (0.008, 0.008, 0.008))
    A2 = Spawn(f"{dir}/WoodenCrate_A2.usd", (0.0065, 0.0065, 0.0065))
    B1 = Spawn(f"{dir}/WoodenCrate_B1.usd", (0.008, 0.008, 0.008))
    B2 = Spawn(f"{dir}/WoodenCrate_B2.usd", (0.0065, 0.0065, 0.0065))
    D1 = Spawn(f"{dir}/WoodenCrate_D1.usd", (0.006, 0.006, 0.006))
    D1_s = Spawn(f"{dir}/WoodenCrate_D1.usd", (0.0045, 0.0045, 0.0045))