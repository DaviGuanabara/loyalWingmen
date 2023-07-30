import numpy as np
import pybullet as p
from modules.models.drone import Drone


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
        
        #velocity_intensity = np.linalg.norm(action)
        #if velocity_intensity != 0:
        #    velocity_unitary_vector = action / velocity_intensity

        #else:
        #   velocity_unitary_vector = np.zeros(3)
        
        #TODO: check if this is correct
        ''''
        
        freq: int=240,
        aggregate_phy_steps: int=1,
        
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        
        temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3], # same as the current position
                                                    target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                    target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
                                                    
                                                    
        '''

        speed_limit = self.informations.speed_limit

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
            target_vel=5 * speed_limit * action,
        )

        return rpm

    # =================================================================================================================
    # Public
    # =================================================================================================================

    def apply_velocity_action(self, action, only_velocity=True):
        if only_velocity:
            p.resetBaseVelocity(self.id, linearVelocity=action)
            #(self.id, action)
        else:    
            rpm = self.__preprocessAction(action)
            self.physics(rpm)
