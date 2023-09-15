# import sys
# sys.path.append("../..")

import math
import numpy as np
from typing import Tuple, List, Dict
import pybullet as p
from gymnasium import spaces
from .sensor_interface import Sensor
from ..dataclasses.quadcopter_specs import QuadcopterSpecs
from ....environments.helpers.environment_parameters import EnvironmentParameters

"""
Ciclo de pegar o pybullet, que seria atualizar e depois pegar os dados
"""


class InertialMeasurementUnit(Sensor):
    def __init__(
        self, parent_id, client_id, environment_parameters: EnvironmentParameters
    ):
        self.parent_id = parent_id
        self.client_id = client_id
        self.environment_parameters = environment_parameters

        timestep = environment_parameters.timestep
        aggregate_physics_step = environment_parameters.aggregate_physics_steps

        self.cycle_duration = timestep * aggregate_physics_step

        self.reset()
        self.update_data()

    def reset(self):
        self.position: np.ndarray = np.zeros(3)
        self.quaternions: np.ndarray = np.zeros(4)
        self.angular_position: np.ndarray = np.zeros(3)

        self.velocity: np.ndarray = np.zeros(3)
        self.angular_velocity: np.ndarray = np.zeros(3)
        self.acceleration: np.ndarray = np.zeros(3)
        self.angular_acceleration: np.ndarray = np.zeros(3)

    def update_data(self):
        # Fetch data from PyBullet
        raw_position, raw_quaternions = p.getBasePositionAndOrientation(
            self.parent_id, physicsClientId=self.client_id
        )
        raw_angular_position = p.getEulerFromQuaternion(raw_quaternions)
        raw_velocity, raw_angular_velocity = p.getBaseVelocity(
            self.parent_id, physicsClientId=self.client_id
        )

        # Convert to numpy arrays
        position = np.array(raw_position)
        quaternions = np.array(raw_quaternions)
        angular_position = np.array(raw_angular_position)
        velocity = np.array(raw_velocity)
        angular_velocity = np.array(raw_angular_velocity)

        # Calculate accelerations
        acceleration = (velocity - self.velocity) / self.cycle_duration
        angular_acceleration = (
            angular_velocity - self.angular_velocity
        ) / self.cycle_duration

        # Update object attributes
        self.position = position
        self.quaternions = quaternions
        self.angular_position = angular_position
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration
        self.angular_acceleration = angular_acceleration

    def read_data(
        self,
    ) -> Dict:
        position = self.position
        attitude = self.angular_position
        quaternions = self.quaternions
        velocity = self.velocity
        angular_rate = self.angular_velocity
        acceleration = self.acceleration
        angular_acceleration = self.angular_acceleration

        return {
            "position": position,
            "attitude": attitude,
            "quaternions": quaternions,
            "velocity": velocity,
            "angular_rate": angular_rate,
            "acceleration": acceleration,
            "angular_acceleration": angular_acceleration,
        }
