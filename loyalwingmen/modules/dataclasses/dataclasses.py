import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field


@dataclass(order=True)
class Kinematics:
    position: np.array = field(default_factory=lambda: np.zeros(3))
    angular_position: np.array = field(default_factory=lambda: np.zeros(3))
    quaternions: np.array = field(default_factory=lambda: np.zeros(4))
    velocity: np.array = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.array = field(default_factory=lambda: np.zeros(3))


@dataclass(order=True)
class Parameters:
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


@dataclass(order=True)
class Informations:
    speed_limit: float = 0
    gravity: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_z_torque: float = 0
    hover_rpm: float = 0
    speed_limit: float = 0
    gnd_eff_h_clip: float = 0
    max_xy_torque: float = 0
