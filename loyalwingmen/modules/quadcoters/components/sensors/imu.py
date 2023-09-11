# import sys
# sys.path.append("../..")

import math
import numpy as np
from typing import Tuple, List, Dict
import pybullet as p
from gymnasium import spaces
from .sensor_interface import Sensor

"""
Ciclo de pegar o pybullet, que seria atualizar e depois pegar os dados
"""


class InertialMeasurementUnit(Sensor):
    def __init__(self, parent_id, client_id):
        self.parent_id = parent_id
        self.client_id = client_id

        self.reset()
        self.update_data()

    def reset(self):
        self.position: np.ndarray = np.zeros(3)
        self.quaternions: np.ndarray = np.zeros(4)
        self.angular_position: np.ndarray = np.zeros(3)

        self.velocity: np.ndarray = np.zeros(3)
        self.angular_velocity: np.ndarray = np.zeros(3)
        
    def update_data(self):
        position, quaternions = p.getBasePositionAndOrientation(
            self.parent_id, physicsClientId=self.client_id
        )
        
        angular_position = p.getEulerFromQuaternion(quaternions)
        velocity, angular_velocity = p.getBaseVelocity(
            self.parent_id, physicsClientId=self.client_id
        )

        self.position = np.array(position)
        self.quaternions = np.array(quaternions)
        self.angular_position = np.array(angular_position)

        self.velocity = np.array(velocity)
        self.angular_velocity = np.array(angular_velocity)

    def read_data(
        self,
    ) -> Dict:
        position = self.position
        attitude = self.angular_position
        quaternions = self.quaternions
        velocity = self.velocity
        angular_rate = self.angular_velocity

        return {
            "position": position,
            "attitude": attitude,
            "quaternions": quaternions,
            "velocity": velocity,
            "angular_rate": angular_rate,
        }
