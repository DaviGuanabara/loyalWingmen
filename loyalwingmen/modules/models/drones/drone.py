import numpy as np
import pybullet as p

from typing import Dict

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
from modules.models.interfaces.sensor_interface import Sensor


#TODO: put here others physics implementations

class DronePhysics:
    def physics(self, rpm: np.ndarray, drone_id: int, drone_parameters: Parameters, client_id: int):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position
        """

        KF = drone_parameters.KF
        KM = drone_parameters.KM

        forces = np.array(rpm**2) * KF
        torques = np.array(rpm**2) * KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                drone_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=client_id,
            )

        p.applyExternalTorque(
            drone_id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=client_id,
        )
        
        
    def apply_forces(self, drone_id, forces):
        # logic to apply forces
        pass




class Drone():
    def __init__(
        self,
        id: int,
        model: DroneModel,
        parameters: Parameters,
        kinematics: Kinematics,
        informations: Informations,
        control: DSLPIDControl,
        environment_parameters: EnvironmentParameters,

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


        
        if self.debug:
            print("Drone created", "debug", environment_parameters.debug)
            
        self.sensors: Dict[str, Sensor] = dict()    
            
        
    # =================================================================================================================
    # Private
    # =================================================================================================================




    


    def replace(self, position: np.ndarray, quaternion: np.ndarray):
        p.resetBasePositionAndOrientation(self.id, position, quaternion)
        self.update_sensors()    

    # =================================================================================================================
    # Sensors
    # =================================================================================================================
    
    def register_sensor(self, sensor_name: str, sensor_instance: Sensor):
        self.sensors[sensor_name] = sensor_instance
        
    def unregister_sensor(self, sensor_name: str):
        if sensor_name in self.sensors:
            del self.sensors[sensor_name]    
    
    def update_sensors(self):
        for sensor_name in self.sensors.keys():
            self.sensors[sensor_name].update_data()
            
    def reset_sensors(self):
        for sensor_name in self.sensors.keys():
            self.sensors[sensor_name].reset()        
            
    def read_sensors(self) -> dict:
        readings = dict()
        
        for sensor_name in self.sensors.keys():
            readings[sensor_name] = self.sensors[sensor_name].read_data()

        return readings