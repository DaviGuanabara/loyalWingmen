import numpy as np
import pybullet as p
import pybullet_data

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple

from utils.factories.agent_factory import Kinematics
from utils.factories.agent_factory import Drone

from control.DSLPIDControl import DSLPIDControl
from utils.factories.agent_factory import gen_drone

################################################################################
## Kinematics
################################################################################

# TODO Manager como Decorator ?


def store_kinematics(client_id, gadget, kinematics):
    assert hasattr(gadget, "kinematics"), "base_manager: gadget dont have 'kinematics'"

    gadget.kinematics = kinematics


def update_kinematics(client_id, gadget):
    assert hasattr(gadget, "kinematics"), "base_manager: gadget dont have 'kinematics'"

    kinematics = collect_kinematics(client_id, gadget)
    store_kinematics(client_id, gadget, kinematics)


def collect_kinematics(client_id, gadget):
    assert hasattr(gadget, "id"), "base_manager: gadget dont have 'id'"

    position, quaternions = p.getBasePositionAndOrientation(
        gadget.id, physicsClientId=client_id
    )
    angular_position = p.getEulerFromQuaternion(quaternions)
    velocity, angular_velocity = p.getBaseVelocity(gadget.id, physicsClientId=client_id)

    kinematics = Kinematics(
        position=np.array(position),
        angular_position=np.array(angular_position),
        quaternions=np.array(quaternions),
        velocity=np.array(velocity),
        angular_velocity=np.array(angular_velocity),
    )

    return kinematics


def _physics():
    raise NotImplementedError
