from enum import Enum

from attr import dataclass

import os.path as osp

from omni_drones.robots import ASSET_PATH

wooden_dir = ASSET_PATH + "/industry_usd/RackLarge/Containers/Wooden"
cardboard_dir = ASSET_PATH + "/industry_usd/RackLarge/Containers/Cardboard"

@dataclass
class PayloadInfo:
    usd_path: str
    scale: tuple[float, float, float]
    shadow_scale: tuple[float, float, float]
    target_pos: tuple[float, float, float]
    target_rot: tuple[float, float, float, float]
    name: str

class Payload(Enum):
    A1 = PayloadInfo(   f"{wooden_dir}/WoodenCrate_A1.usd", (0.01, 0.01, 0.01),      (0.5, 0.5, 0.5),        (-7., -7, 0),   (1., 0, 0, 0), 'A1')
    A2 = PayloadInfo(   f"{wooden_dir}/WoodenCrate_A2.usd", (0.005, 0.005, 0.005),   (0.25, 0.25, 0.25),  (3., 5, 0),     (1., 0, 0, 0), 'A2')
    B1 = PayloadInfo(   f"{wooden_dir}/WoodenCrate_B1.usd", (0.01, 0.01, 0.01),      (0.5, 1.0, 0.5),        (7., -7., 0),    (1., 0, 0, 0), 'B1')
    B2 = PayloadInfo(   f"{wooden_dir}/WoodenCrate_B2.usd", (0.005, 0.005, 0.005),   (0.25, 0.5, 0.25),   (-3., 5, 0),    (1., 0, 0, 0), 'B2')
    D1 = PayloadInfo(   f"{wooden_dir}/WoodenCrate_D1.usd", (0.006, 0.006, 0.006),      (0.6, 0.9, 0.3),        (-7., 7., 0),    (1., 0, 0, 0), 'D1')
    D1_s = PayloadInfo( f"{wooden_dir}/WoodenCrate_D1.usd", (0.003, 0.003, 0.003),   (0.3, 0.45, 0.15),   (0., -5, 0),    (1., 0, 0, 0), 'D1_s')
    # CA1 = PayloadInfo(  f"{cardboard_dir}/Cardbox_A1.usd",      (0.015, 0.015, 0.015),      (0.4, 0.4, 0.4),        (-1., -2, 0),   (1., 0, 0, 0), 'CA1')
    # CA2 = PayloadInfo(  f"{cardboard_dir}/Cardbox_A2.usd",      (0.012, 0.012, 0.012),       (0.325, 0.325, 0.325),  (1., 2, 0),     (1., 0, 0, 0), 'CA2')
    # CB1 = PayloadInfo(  f"{cardboard_dir}/Cardbox_B1.usd",     (0.012, 0.03, 0.016),      (0.4, 0.8, 0.4),        (0., -2, 0),    (1., 0, 0, 0), 'CB1')
    # CB2 = PayloadInfo(  f"{cardboard_dir}/Cardbox_B2.usd",     (0.010, 0.025, 0.014),   (0.325, 0.65, 0.325),   (-1., 2, 0),    (1., 0, 0, 0), 'CB2')
    # CC1 = PayloadInfo(  f"{cardboard_dir}/Cardbox_C1.usd",     (0.023, 0.035, 0.024),      (0.6, 0.9, 0.3),        (0., 3., 0),    (1., 0, 0, 0), 'CC1')
    # CC2 = PayloadInfo(  f"{cardboard_dir}/Cardbox_C2.usd",     (0.018, 0.026, 0.018),   (0.45, 0.675, 0.225),   (0., -2, 0),    (1., 0, 0, 0), 'CC2')



    @staticmethod
    def get(i):
        for _i, member in enumerate(Payload):
            if i == _i:
                return member
