import numpy as np
import pybullet as p
from modules.models.drone import Drone, ObservationType
from dataclasses import dataclass, field, fields

class LoyalWingman(Drone):

    # =================================================================================================================
    # Private
    # =================================================================================================================

    # def __init__(self):
    #    super().__init__()

    def __preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """Pre-processes the velocty action into motors' RPMs.
        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.
        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.
        """
        
        
        speed_limit = self.informations.speed_limit #speed_limit == 8.333333333333334
        speed_amplification = self.informations.speed_amplification #speed_amplification == 20

        position: np.ndarray = self.kinematics.position
        quaternions: np.ndarray = self.kinematics.quaternions
        angular_position: np.ndarray = self.kinematics.angular_position

        velocity: np.ndarray = self.kinematics.velocity
        angular_velocity: np.ndarray = self.kinematics.angular_velocity

        
        control_timestep = self.environment_parameters.aggregate_physics_steps * self.environment_parameters.timestep_period
        rpm, _, _ = self.control.computeControl(
            control_timestep=control_timestep,
            cur_pos=position,
            cur_quat=quaternions,
            cur_vel=velocity,
            cur_ang_vel=angular_velocity,
            target_pos=position,
            target_rpy=angular_position,
            target_vel=speed_amplification * speed_limit * action,
        )

        return rpm

    # =================================================================================================================
    # Public
    # =================================================================================================================

    def apply_velocity_action(self, velocity: np.ndarray, only_velocity=True):
        if only_velocity:
            
            #Zera alguma velocidade prévia
            #p.resetBaseVelocity(self.id, linearVelocity=np.array([0, 0, 0]), angularVelocity=np.array([0, 0, 0]), physicsClientId=self.client_id)
            
            speed_limit = self.informations.speed_limit #speed_limit == 8.333333333333334
            #speed_amplification = self.informations.speed_amplification #speed_amplification == 20
            speed_amplification = self.informations.speed_amplification
            target_vel=speed_amplification * speed_limit * velocity
            
            
            weigth = self.environment_parameters.G * self.parameters.M
            self.apply_force(np.array([0, 0, weigth]))
            self.apply_velocity(velocity=target_vel, angular_velocity=np.array([0, 0, 0]))
            
        else:    
            rpm = self.__preprocessAction(velocity)
            self.physics(rpm)
    
    def apply_frozen_behavior(self):
        weigth = self.environment_parameters.G * self.parameters.M
        self.apply_force(np.array([0, 0, weigth]))
        self.apply_velocity(velocity=np.array([0, 0, 0]), angular_velocity=np.array([0, 0, 0]))

    def apply_constant_velocity_behavior(self):
        self.apply_frozen_behavior()
        self.apply_velocity(
            velocity=np.array([.1, .1, 0]), angular_velocity=np.array([0, 0, 0]))

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
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ):
        p.resetBaseVelocity(
            self.id,
            velocity,
            angular_velocity,
            physicsClientId=self.environment_parameters.client_id,
        )
