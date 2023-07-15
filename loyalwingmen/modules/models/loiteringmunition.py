import numpy as np
import pybullet as p
from modules.models.drone import Drone


################################################################################
# Action
################################################################################


class LoiteringMunition(Drone):

    # def __init__(self):
    #    super().__init__()

    def apply_frozen_behavior(self):
        weigth = self.environment_parameters.G * self.parameters.M
        self.apply_force(np.array([0, 0, weigth]))
        self.apply_velocity(velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

    def apply_constant_velocity_behavior(self):
        self.apply_frozen_behavior()
        self.apply_velocity(
            velocity=[.1, .1, 0], angular_velocity=[0, 0, 0])

    def apply_force(self, force):
        p.applyExternalForce(
            self.id,
            -1,
            forceObj=force,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.environment_parameters.client_id,
        )

    def apply_velocity(
        self,
        velocity: np.array,
        angular_velocity: np.array,
    ):
        p.resetBaseVelocity(
            self.id,
            velocity,
            angular_velocity,
            physicsClientId=self.environment_parameters.client_id,
        )

