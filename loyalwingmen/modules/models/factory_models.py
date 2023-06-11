import numpy as np
import pybullet as p
import pybullet_data

from modules.control.DSLPIDControl import DSLPIDControl
from modules.utils.enums import DroneModel, Physics, ImageType

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image

from modules.behaviors.tree.behavior_tree_base import BehaviorTree


from dataclasses import dataclass, field
from typing import NamedTuple

# Mutável e Imutável
# https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
# bar: list = field(default_factory=list)
# from dataclasses import dataclass, field


@dataclass
class Kinematics:
    position: np.array = field(default_factory=lambda: np.zeros(3))
    angular_position: np.array = field(default_factory=lambda: np.zeros(3))
    quaternions: np.array = field(default_factory=lambda: np.zeros(4))
    velocity: np.array = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.array = field(default_factory=lambda: np.zeros(3))


class Parameters(NamedTuple):
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: float
    J_INV: float
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: float
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float


class Drone_Informations:
    speed_limit: float = 0
    gravity: float = 0
    max_rpm: float = 0
    max_thrust: float = 0

    max_z_torque: float = 0
    hover_rpm: float = 0
    speed_limit: float = 0
    gnd_eff_h_clip: float = 0

    max_xy_torque: float = 0


class Drone:
    sort_index: int = field(init=False, repr=False)
    id: int
    parameters: Parameters
    kinematics: Kinematics
    informations: Drone_Informations
    control: DSLPIDControl = DSLPIDControl(drone_model=DroneModel.CF2X)

    def __post_init__(self):
        """
        initializes after init
        """
        self.sort_index = self.id


class Obstacle_Informations:
    mass: float = 0


class Obstacle:  # Imutável com (NamedTuple):
    id: int
    kinematics: Kinematics
    informations: Obstacle_Informations
