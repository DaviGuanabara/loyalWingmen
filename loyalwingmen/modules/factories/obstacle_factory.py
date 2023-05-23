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
from modules.factories.factory_models import Kinematics
from modules.factories.factory_models import Obstacle_Informations, Obstacle
from modules.decorators.obstacle_decorator import ObstacleDecorator


class ObstacleFactory:
    def __init__(self):
        pass

    def generate_extended_obstacle(
        self,
        environment_parameters,  # client_id: int,
        urdf_file_path: str,
        initial_position: np.array = np.ones(3),
        initial_angular_position: np.array = np.zeros(3),
    ):
        kinematics = Kinematics(
            position=initial_position, angular_position=initial_angular_position
        )

        obstacle_id = self.load_obstacle(
            environment_parameters.client_id,
            urdf_file_path,
            initial_position,
            initial_angular_position,
        )

        obstacle_informations = Obstacle_Informations()
        obstacle_informations.mass = p.getDynamicsInfo(
            obstacle_id, -1, physicsClientId=environment_parameters.client_id
        )[0]

        obstacle = Obstacle()
        obstacle.id = obstacle_id
        obstacle.kinematics = kinematics
        obstacle.informations = obstacle_informations

        return ObstacleDecorator(environment_parameters, obstacle)

    def load_obstacle(
        self,
        client_id: int,
        urdf_file_path: str = "cube_small.urdf",
        initial_position: np.array = np.ones(3),
        initial_angular_position: np.array = np.zeros(3),
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
