import numpy as np
import pybullet as p


from modules.interfaces.drone_interface import IDrone

from modules.models.lidar import LiDAR
from modules.control import DSLPIDControl
from modules.environments.environment_models import EnvironmentParameters
from modules.dataclasses.dataclasses import Parameters, Kinematics, Informations


class Drone(IDrone):

    def __init__(self, id: int, client_id: int, parameters: Parameters, kinematics: Kinematics, informations: Informations, control: DSLPIDControl, environment_parameters: EnvironmentParameters):
        super().__init__(id, client_id, parameters, kinematics,
                         informations, control, environment_parameters)

        self.set_lidar_parameters(max_distance=3, resolution=1)

    # =================================================================================================================
    # Private
    # =================================================================================================================

    def __physics(self, rpm: np.array):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position
        """

        KF = self.parameters.KF
        KM = self.parameters.KM

        forces = np.array(rpm**2) * KF
        torques = np.array(rpm**2) * KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )

        p.applyExternalTorque(
            self.id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

    def __collect_kinematics(self) -> Kinematics:
        position, quaternions = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client_id
        )

        angular_position = p.getEulerFromQuaternion(quaternions)
        velocity, angular_velocity = p.getBaseVelocity(
            self.id, physicsClientId=self.client_id
        )

        kinematics = Kinematics(
            position=np.array(position),
            angular_position=np.array(angular_position),
            quaternions=np.array(quaternions),
            velocity=np.array(velocity),
            angular_velocity=np.array(angular_velocity),
        )

        return kinematics

    # =================================================================================================================
    # Public
    # =================================================================================================================

    def store_kinematics(self, kinematics: Kinematics):
        self.kinematics = kinematics

    def update_kinematics(self):
        kinematics = self.__collect_kinematics()
        self.store_kinematics(kinematics)

    def set_lidar_parameters(self, max_distance: int = 3, resolution: int = 1):
        self.lidar: LiDAR = LiDAR(
            max_distance=max_distance, resolution=resolution)

    def observation(self, loitering_munition_position: np.array = np.array([]), obstacle_position: np.array = np.array([]), loyalwingman_position: np.array = np.array([]), current_position: np.array = np.array([0, 0, 0])):
        self.lidar.add_position(loitering_munition_position=loitering_munition_position, obstacle_position=obstacle_position,
                                loyalwingman_position=loyalwingman_position, current_position=current_position)
