import numpy as np
import pybullet as p

from loyalwingmen.modules.environments.dataclasses.environment_parameters import EnvironmentParameters
from loyalwingmen.modules.quadcoters.base.quadcopter import QuadcopterType
from loyalwingmen.modules.quadcoters.components.dataclasses.operational_constraints import OperationalConstraints
from loyalwingmen.modules.quadcoters.components.dataclasses.quadcopter_specs import QuadcopterSpecs
from loyalwingmen.modules.utils.enums import DroneModel
from .base.quadcopter import Quadcopter
from dataclasses import dataclass, field, fields

class LoyalWingman(Quadcopter):

    def __init__(self, id: int, model: DroneModel, droneSpecs: QuadcopterSpecs, operationalConstraints: OperationalConstraints, environment_parameters: EnvironmentParameters, quadcopter_name: str, use_direct_velocity: bool = False):
        super().__init__(id, model, droneSpecs, operationalConstraints, environment_parameters, QuadcopterType.LOYALWINGMAN, use_direct_velocity)
        self.quadcopter_name = quadcopter_name
        