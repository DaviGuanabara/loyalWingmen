import numpy as np
import pybullet as p
import pybullet_data

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple

from modules.factories.agent_factory import Kinematics
from modules.factories.agent_factory import Drone

from modules.control.DSLPIDControl import DSLPIDControl
from modules.factories.agent_factory import gen_drone
import modules.factories.obstacle_factory
from modules.managers.base_manager import (
    collect_kinematics,
    update_kinematics,
    store_kinematics,
)
from modules.factories.obstacle_factory import Obstacle, generate_obstacle


################################################################################
## Action
################################################################################


def apply_frozen_behavior(
    environment_parameters,
    obstacle: Obstacle,
):
    obstacle_weigth = environment_parameters.G * obstacle.informations.mass

    apply_force(environment_parameters, obstacle, np.array([0, 0, obstacle_weigth]))
    apply_velocity(environment_parameters, obstacle, [0, 0, 0], [0, 0, 0])


def apply_force(environment_parameters, obstacle: Obstacle, force):
    p.applyExternalForce(
        obstacle.id,
        -1,
        forceObj=force,
        posObj=[0, 0, 0],
        flags=p.LINK_FRAME,
        physicsClientId=environment_parameters.client_id,
    )


def apply_velocity(
    environment_parameters, obstacle, velocity: np.array, angular_velocity: np.array
):
    p.resetBaseVelocity(
        obstacle.id,
        velocity,
        angular_velocity,
        physicsClientId=environment_parameters.client_id,
    )
