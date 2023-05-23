import numpy as np
import pybullet as p
import pybullet_data

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple

from modules.factories.factory_models import Kinematics
from modules.factories.factory_models import Drone

from modules.control.DSLPIDControl import DSLPIDControl


################################################################################
## Kinematics
################################################################################

# TODO Manager como Decorator ?


class BaseDecorator:
    def __init__(self, client_id, gadget):
        self.client_id = client_id
        self.gadget = gadget
        pass

    def store_kinematics(self, kinematics):
        self.client_id
        self.gadget
        assert hasattr(
            self.gadget, "kinematics"
        ), "base_manager: gadget dont have 'kinematics'"

        self.gadget.kinematics = kinematics

    def update_kinematics(self):
        assert hasattr(
            self.gadget, "kinematics"
        ), "base_manager: gadget dont have 'kinematics'"

        kinematics = self.collect_kinematics()
        self.store_kinematics(kinematics)

    def collect_kinematics(self):
        assert hasattr(self.gadget, "id"), "base_manager: gadget dont have 'id'"

        position, quaternions = p.getBasePositionAndOrientation(
            self.gadget.id, physicsClientId=self.client_id
        )
        angular_position = p.getEulerFromQuaternion(quaternions)
        velocity, angular_velocity = p.getBaseVelocity(
            self.gadget.id, physicsClientId=self.client_id
        )

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
