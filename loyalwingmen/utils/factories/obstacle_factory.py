import numpy as np
import pybullet as p
import pybullet_data

from control.DSLPIDControl import DSLPIDControl
from utils.enums import DroneModel, Physics, ImageType

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass, field
from typing import NamedTuple

#TODO fazer um arquivo chamado base_factory com os dataclasses e as funções bases
# load_agent
# generator

def load_agent(
    client_id: int,
    urdf_file_path: str,
    initial_position=np.ones(3),
    initial_angular_position=np.zeros(3),
):
    return p.loadURDF(
        urdf_file_path,
        initial_position,
        p.getQuaternionFromEuler(initial_angular_position),
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client_id,
    )


def gen_drone(
    client_id: int,
    urdf_file_path: str,
    initial_position: np.array = np.ones(3),
    initial_angular_position: np.array = np.zeros(3),
    gravity_acceleration: float = 9.8,
):
    parameters = _parseURDFParameters(
        urdf_file_path
    )  # TODO mudar o nome, pois ele é específico do drone, e não é genérico, como o nome dá a entender.

    print(initial_position)
    print(initial_angular_position)
    kinematics = Kinematics(
        position=initial_position, 
        angular_position=initial_angular_position
    )
    id = load_agent(
        client_id, urdf_file_path, initial_position, initial_angular_position
    )
    informations = compute_informations(
        parameters, gravity_acceleration=gravity_acceleration
    )

    drone = Drone()
    drone.id = id
    drone.parameters = parameters
    drone.kinematics = kinematics
    drone.informations = informations

    return drone
