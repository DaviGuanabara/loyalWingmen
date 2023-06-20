import numpy as np
from modules.models.drone import Drone


class LoyalWingman(Drone):

    # =================================================================================================================
    # Private
    # =================================================================================================================

    # def __init__(self):
    #    super().__init__()

    def __preprocessAction(self, action: np.array) -> np.array:
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

        speed_limit = self.informations.speed_limit

        position: np.array = self.kinematics.position
        quaternions: np.array = self.kinematics.quaternions
        angular_position: np.array = self.kinematics.angular_position

        velocity: np.array = self.kinematics.velocity
        angular_velocity: np.array = self.kinematics.angular_velocity

        if np.linalg.norm(action[:3]) != 0:
            velocity_unitary_vector = action[:3] / np.linalg.norm(action[:3])

        else:
            velocity_unitary_vector = np.zeros(3)

        rpm, _, _ = self.control.computeControl(
            control_timestep=self.environment_parameters.timestep_period,
            cur_pos=position,
            cur_quat=quaternions,
            cur_vel=velocity,
            cur_ang_vel=angular_velocity,
            target_pos=position,
            target_rpy=angular_position,
            target_vel=speed_limit *
            np.abs(action[3]) * velocity_unitary_vector,
        )

        return rpm

    # =================================================================================================================
    # Public
    # =================================================================================================================

    def apply_velocity_action(self, action):
        rpm = self.__preprocessAction(action)
        self.physics(rpm)
