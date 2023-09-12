import numpy as np
import pybullet as p

from .components.base.quadcopter import Quadcopter, QuadcopterType, FlightStateManager, QuadcopterSpecs, OperationalConstraints, EnvironmentParameters, DroneModel
from dataclasses import dataclass, field, fields

class LoyalWingman(Quadcopter):

    def __init__(self, id: int, model: DroneModel, droneSpecs: QuadcopterSpecs, operationalConstraints: OperationalConstraints, environment_parameters: EnvironmentParameters, quadcopter_name: str, use_direct_velocity: bool = False):
        super().__init__(id, model, droneSpecs, operationalConstraints, environment_parameters, QuadcopterType.LOYALWINGMAN, quadcopter_name, use_direct_velocity)
        self.quadcopter_name = quadcopter_name
        