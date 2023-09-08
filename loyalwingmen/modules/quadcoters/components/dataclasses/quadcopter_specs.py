from dataclasses import dataclass
import numpy as np


@dataclass(order=True)
class QuadcopterSpecs:
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: np.ndarray
    J_INV: np.ndarray
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: np.ndarray
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float
    WEIGHT: float
