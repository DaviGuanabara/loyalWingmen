import numpy as np
import pybullet as p
from typing import Dict
from ..dataclasses.flight_state import FlightState
from ..dataclasses.quadcopter_specs import QuadcopterSpecs
from .actuator_interface import ActuatorInterface


class Propulsion(ActuatorInterface):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        self.drone_specs = drone_specs
        self.client_id = client_id
        self.drone_id = drone_id

    def apply(self, action):
        assert False, "Not implemented"


class Motors(Propulsion):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        super().__init__(drone_id, drone_specs, client_id)

    def apply(
        self,
        rpm: np.ndarray,
    ):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position
        """

        KF = self.drone_specs.KF
        KM = self.drone_specs.KM

        forces = np.array(rpm**2) * KF
        torques = np.array(rpm**2) * KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.drone_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )

        p.applyExternalTorque(
            self.drone_id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )


class DirectVelocityApplier(Propulsion):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        super().__init__(drone_id, drone_specs, client_id)

    def apply(
        self,
        velocity: np.ndarray,
        angularVelocity: np.ndarray = np.zeros(3),
    ):
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=velocity,
            angularVelocity=angularVelocity,
            physicsClientId=self.client_id,
        )
