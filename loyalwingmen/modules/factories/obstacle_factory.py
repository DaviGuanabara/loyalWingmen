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
from modules.factories.base_factory import Kinematics


class Obstacle_Informations:
    mass: float = 0


class Obstacle:  # Imutável com (NamedTuple):
    id: int
    kinematics: Kinematics
    informations: Obstacle_Informations


def load_obstacle(
    client_id: int,
    urdf_file_path: str = "cube_small.urdf",
    initial_position: np.array = np.ones(3),
    initial_angular_position: np.array = np.zeros(3),
    urdf: str = "cube_small.urdf",
):
    """Add obstacles to the environment.
    These obstacles are loaded from standard URDF files included in Bullet.
    """

    return p.loadURDF(
        urdf_file_path,
        initial_position,
        p.getQuaternionFromEuler(initial_angular_position),
        physicsClientId=client_id,
    )


def generate_obstacle(
    client_id: int,
    urdf_file_path: str,
    initial_position: np.array = np.ones(3),
    initial_angular_position: np.array = np.zeros(3),
):
    kinematics = Kinematics(
        position=initial_position, angular_position=initial_angular_position
    )

    obstacle_id = load_obstacle(
        client_id, urdf_file_path, initial_position, initial_angular_position
    )

    obstacle_informations = Obstacle_Informations()
    obstacle_informations.mass = p.getDynamicsInfo(
        obstacle_id, -1, physicsClientId=client_id
    )[0]

    obstacle = Obstacle()
    obstacle.id = obstacle_id
    obstacle.kinematics = kinematics
    obstacle.informations = obstacle_informations

    return obstacle
