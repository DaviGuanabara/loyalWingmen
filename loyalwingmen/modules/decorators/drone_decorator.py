import numpy as np
import pybullet as p
import pybullet_data

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple

from modules.factories.factory_models import Drone
from modules.decorators.base_decorator import BaseDecorator


class DroneDecorator(BaseDecorator):
    ################################################################################
    ## Action
    ################################################################################
    def __init__(self, drone: Drone, environment_parameters):
        self.drone: Drone = drone
        self.environment_parameters = environment_parameters

        super().__init__(environment_parameters.client_id, drone)
        pass

    # TODO fazer um dataclass chamado PYBULLET PARAMETERS, NO QUAL ARMAZENA OS DADOS ASSOCIADOS AO PYBULLET

    def get(self, attribute: str):
        return self.drone.__getattribute__(attribute)

    def get_kinematics(self):
        return self.drone.kinematics

    def get_control(self):
        return self.drone.control

    def get_informations(self):
        return self.drone.informations

    def get_parameters(self):
        return self.drone.parameters

    def get_id(self):
        return self.drone.id

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.
        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.
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
        drone = self.drone
        speed_limit = drone.informations.speed_limit

        position = drone.kinematics.position
        quaternions = drone.kinematics.quaternions
        angular_position = drone.kinematics.angular_position

        velocity = drone.kinematics.velocity
        angular_velocity = drone.kinematics.angular_velocity

        if np.linalg.norm(action[:3]) != 0:
            v_unit_vector = action[:3] / np.linalg.norm(action[:3])

        else:
            v_unit_vector = np.zeros(3)

        rpm, _, _ = drone.control.computeControl(
            control_timestep=self.environment_parameters.timestep_period,
            cur_pos=np.array(position),
            cur_quat=quaternions,
            cur_vel=velocity,
            cur_ang_vel=angular_velocity,
            # same as the current position
            target_pos=np.array(position),
            target_rpy=[
                0,
                0,
                angular_position[2],
            ],  # angular_position,  # keep current yaw
            # target the desired velocity vector
            target_vel=speed_limit * np.abs(action[3]) * v_unit_vector,
        )

        return rpm

    # TODO: Preciso remover esse freq aí, e colocar o do ambiente.
    def apply_velocity_action(self, action):
        drone = self.drone
        rpm = self._preprocessAction(action)
        self._physics(rpm)

    def _physics(self, rpm):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """

        drone = self.drone
        drone_id = drone.id
        KF = drone.parameters.KF
        KM = drone.parameters.KM

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
                physicsClientId=self.client_id,
            )
        p.applyExternalTorque(
            drone_id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

    def execute_behavior(self):
        # raise NotImplementedError
        # drone = self.drone
        # drone.behavior.update(drone)
        # self.drone.
        action = [0.2, 0.1, -0.1, 0.01]
        # self.apply_velocity_action([0.2, 0.1, -0.1, 0.01])
        pass

        return action
