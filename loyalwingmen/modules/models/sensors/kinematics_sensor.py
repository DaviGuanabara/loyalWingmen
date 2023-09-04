import sys
sys.path.append("../..")

import math
import numpy as np
from typing import Tuple, List
import pybullet as p
from gymnasium import spaces
from modules.models.interfaces.sensor_interface import Sensor

"""
Ciclo de pegar o pybullet, que seria atualizar e depois pegar os dados
"""
class KinematicSensor(Sensor):
    def __init__(self, parent_id, client_id):
        self.parent_id = parent_id
        self.client_id = client_id
        
        self.position = np.zeros(3)
        self.quaternions = np.zeros(4)
        self.angular_position = np.zeros(3)
        
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def update_date(self):
        position, quaternions = p.getBasePositionAndOrientation(
            self.parent_id , physicsClientId=self.client_id
        )
        angular_position = p.getEulerFromQuaternion(quaternions)
        velocity, angular_velocity = p.getBaseVelocity(
            self.parent_id , physicsClientId=self.client_id
        )
        
        self.position = position
        self.quaternions = quaternions
        self.angular_position = angular_position
        
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def read_data(self) -> np.ndarray:
        position = self.position
        angular_position = self.angular_position
        quaternions = self.quaternions
        velocity = self.velocity
        angular_velocity = self.angular_velocity
        
        return np.concatenate([position, angular_position, quaternions, velocity, angular_velocity], axis=0)
