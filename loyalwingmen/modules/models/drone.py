import numpy as np
import pybullet as p


from modules.interfaces.drone_interface import IDrone

from modules.models.lidar import LiDAR
from modules.control import DSLPIDControl
from modules.environments.environment_models import EnvironmentParameters
from modules.dataclasses.dataclasses import Parameters, Kinematics, Informations
from modules.utils.enums import DroneModel


class Drone(IDrone):

    def __init__(self, id: int, model: DroneModel, parameters: Parameters, kinematics: Kinematics, informations: Informations, control: DSLPIDControl, environment_parameters: EnvironmentParameters):
        self.id: int = id
        self.client_id: int = environment_parameters.client_id
        self.model = model
        self.parameters: Parameters = parameters
        self.kinematics: Kinematics = kinematics
        self.informations: Informations = informations
        self.control: DSLPIDControl = control
        self.environment_parameters: EnvironmentParameters = environment_parameters
        self.__setup()

    def __setup(self):
        """This function is called in init
        Parameters
        ----------
        Returns
        ----------
        """
        resolution: float = 1
        radius: float = 5
        self.set_lidar_parameters(radius, resolution)

    # =================================================================================================================
    # Private
    # =================================================================================================================

    def physics(self, rpm: np.array):
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

    def collect_kinematics(self) -> Kinematics:
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
        kinematics = self.collect_kinematics()
        self.store_kinematics(kinematics)

    def set_lidar_parameters(self, radius: float = 5, resolution: float = 1):
        print("lidar setup")
        self.lidar: LiDAR = LiDAR(radius, resolution)
        print("done")

    def observation(self, loyalwingmen: np.array = np.array([], dtype=IDrone), loitering_munitions: np.array = np.array([], dtype=IDrone), obstacles: np.array = np.array([])):
        self.lidar.reset()

        for lw in loyalwingmen:
            self.lidar.add_position(
                loyalwingman_position=lw.kinematics.position, current_position=self.kinematics.position)

        for lm in loitering_munitions:
            self.lidar.add_position(
                loitering_munition_position=lm.kinematics.position, current_position=self.kinematics.position)

        for obstacle in obstacles:
            self.lidar.add_position(
                obstacle_position=obstacle.kinematics.position, current_position=self.kinematics.position)

        return self.lidar.get_sphere()
