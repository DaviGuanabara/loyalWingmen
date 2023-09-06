import numpy as np
from dataclasses import dataclass

@dataclass(order=True)
class OperationalConstraints:
    speed_limit: float = 0
    speed_amplification: float = 0
    gravity: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_z_torque: float = 0
    hover_rpm: float = 0
    gnd_eff_h_clip: float = 0
    max_xy_torque: float = 0