import numpy as np
import pybullet as p


from modules.interfaces.drone_interface import IDrone

from modules.models.lidar import LiDAR
from modules.control.DSLPIDControl import DSLPIDControl
from modules.dataclasses.dataclasses import (
    Parameters,
    Kinematics,
    Informations,
    EnvironmentParameters,
)
from dataclasses import dataclass, field
from modules.utils.enums import DroneModel
from typing import List, TYPE_CHECKING, Union, Optional

from enum import Enum


    
if TYPE_CHECKING:
    from modules.factories.loyalwingman_factory import LoyalWingman
    from modules.factories.loiteringmunition_factory import LoiteringMunition

    #from LoyalWingmen import LoyalWingmen
    #from LoiteringMunition import LoiteringMunition
    


class ObservationType(Enum):
    OTHER_SOURCE = 0
    LIDAR = 1
    KINEMATICS = 2
    MIXED_LIDAR_KINEMATICS = 3
    

class Drone(IDrone):
    def __init__(
        self,
        id: int,
        model: DroneModel,
        parameters: Parameters,
        kinematics: Kinematics,
        informations: Informations,
        control: DSLPIDControl,
        environment_parameters: EnvironmentParameters,
        lidar: Union[None, LiDAR] = None,
        observation_type: ObservationType = ObservationType.LIDAR,
    ):
        self.id: int = id
        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug
        
        self.model = model
        self.parameters: Parameters = parameters
        self.kinematics: Kinematics = kinematics
        self.informations: Informations = informations
        self.control: DSLPIDControl = control
        self.environment_parameters: EnvironmentParameters = environment_parameters

        self.setup_observation(observation_type, lidar)
        
        if self.debug:
            print("Drone created", "debug", environment_parameters.debug)
            
    
    def setup_observation(self, observation_type: ObservationType, lidar: Optional[LiDAR] = None):
        
        self.observation_type = observation_type
        
        if observation_type == ObservationType.LIDAR and lidar is not None:
            self.lidar: LiDAR = lidar
            self.OBSERVATION_TYPE = observation_type
            return
        
        if observation_type == ObservationType.MIXED_LIDAR_KINEMATICS:
            self.observation_type = ObservationType.KINEMATICS
            return            

        
    # =================================================================================================================
    # Private
    # =================================================================================================================

    def physics(self, rpm: np.ndarray):
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

    #def set_lidar_parameters(self, radius: float = 5, resolution: float = 1):
        #print(self.debug)
    #    self.lidar: LiDAR = LiDAR(radius, resolution, client_id=self.client_id, debug=self.debug)

    def replace(self, position: np.ndarray, quaternion: np.ndarray):
        p.resetBasePositionAndOrientation(self.id, position, quaternion)
        self.update_kinematics()

    def observation_space(self):
            if ObservationType.LIDAR == self.observation_type:
                return self.lidar.observation_space()
            else:
                return self.kinematics.observation_space()
            
    def observation(self, *args, **kwargs) -> np.ndarray:
        
        """
        Return the observation of the environment.

        Parameters
        ----------
        *args, **kwargs
            Any additional arguments are passed to lidar_observation when self.observation_type == ObservationType.LIDAR is True.
            
        If self.observation_type == ObservationType.LIDAR,
        then the following parameters are passed to lidar_observation:
            loyalwingmen: List["LoyalWingman"] = [],
            loitering_munitions: List["LoiteringMunition"] = [],
            obstacles: List = [],
            
        Otherwise, the following parameters are passed to default_observation:
            None        

        Returns
        -------
        np.ndarray
            The observation array.
        """
        
        
        if self.observation_type == ObservationType.LIDAR:
            return self.lidar_observation(*args, **kwargs)
        else:
            return self.default_observation()

    def lidar_observation(
        self,
        loyalwingmen: List["LoyalWingman"] = [],
        loitering_munitions: List["LoiteringMunition"] = [],
        obstacles: List = [],
    ) -> np.ndarray:
        
        #kwargs hints - object not interable. So, i cannot use this approach.
        #TODO. Fix kward hints, using args.
        self.lidar.reset()

        for lw in loyalwingmen:
            self.lidar.add_position(
                loyalwingman_position=lw.kinematics.position,
                current_position=self.kinematics.position,
            )

        for lm in loitering_munitions:
            self.lidar.add_position(
                loitering_munition_position=lm.kinematics.position,
                current_position=self.kinematics.position,
            )

        for obstacle in obstacles:
            self.lidar.add_position(
                obstacle_position=obstacle.kinematics.position,
                current_position=self.kinematics.position,
            )

        return self.lidar.get_sphere()

    def default_observation(self) -> np.ndarray:
        return self.kinematics.observation()

    def get_observation_features(self) -> List:
        
        if self.observation_type == ObservationType.LIDAR:
            return self.lidar.get_features()
        
        return []
    
    def observation_parameters(self) -> dict:
        parameters = {}
        if self.observation_type == ObservationType.LIDAR:
            parameters = self.lidar.parameters()
            return parameters
        
        return parameters
    