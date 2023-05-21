import numpy as np
import pybullet as p
import pybullet_data

from modules.control.DSLPIDControl import DSLPIDControl
from modules.utils.enums import DroneModel, Physics, ImageType

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class Kinematics:
    position: np.array = field(default_factory=lambda: np.zeros(3))
    angular_position: np.array = field(default_factory=lambda: np.zeros(3))
    quaternions: np.array = field(default_factory=lambda: np.zeros(4))
    velocity: np.array = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.array = field(default_factory=lambda: np.zeros(3))


def load():
    raise NotImplementedError


def generate():
    raise NotImplementedError
