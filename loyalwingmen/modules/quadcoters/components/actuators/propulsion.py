import numpy as np
import pybullet as p
from typing import Dict
from ..dataclasses.flight_state import FlightStateManager
from modules.utils.enums import DroneModel
from ..dataclasses.quadcopter_specs import QuadcopterSpecs
from .actuator_interface import ActuatorInterface


from typing import Dict

from ..controllers.DSLPIDControl import DSLPIDControl

from ..dataclasses.quadcopter_specs import QuadcopterSpecs


from ..dataclasses.flight_state import FlightStateManager
from ....environments.dataclasses.environment_parameters import EnvironmentParameters

from modules.utils.enums import DroneModel
from typing import List, TYPE_CHECKING, Union, Optional

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

        # A força é a plicada, mas não consegue compensar a gravidade não sei o motivo
        # assim, prefiro deixar comentado, sem gravidade, sem compensação.
        # p.applyExternalForce(
        #    self.drone_id,
        #    -1,
        #    forceObj=np.array([0, 0, self.drone_specs.WEIGHT]),
        #    posObj=[0, 0, 0],
        #    flags=p.LINK_FRAME,
        #    physicsClientId=self.client_id,
        # )

        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=velocity,
            angularVelocity=angularVelocity,
            physicsClientId=self.client_id,
        )


class PropulsionSystem:
    def __init__(
        self,
        drone_id: int,
        model: DroneModel,
        droneSpecs: QuadcopterSpecs,
        environment_parameters: EnvironmentParameters,
        quadcopter_name: str = "",
        use_direct_velocity: bool = False,
    ):
        self.quadcopter_name = quadcopter_name
        self.environment_parameters = environment_parameters
        self.use_direct_velocity = use_direct_velocity

        if use_direct_velocity:
            self.direct_velocity_applier = DirectVelocityApplier(
                drone_id=drone_id,
                drone_specs=droneSpecs,
                client_id=environment_parameters.client_id,
            )
        else:
            self.controller = DSLPIDControl(
                model, droneSpecs, environment_parameters
            )  # default or custom controller
            self.motors = Motors(
                drone_id=drone_id,
                drone_specs=droneSpecs,
                client_id=environment_parameters.client_id,
            )

    def _apply_controller(
        self, target_velocity: np.ndarray, flight_state_manager: FlightStateManager
    ):
        inertial_data = flight_state_manager.get_inertial_data()

        yaw = (inertial_data["attitude"] or np.zeros(3))[2]
        target_rpy = np.array([0, 0, yaw])

        aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps
        timestep_period = self.environment_parameters.timestep_period
        control_timestep = aggregate_physics_steps * timestep_period

        rpm, _, _ = self.controller.computeControl(
            control_timestep,
            inertial_data["position"],
            inertial_data["quaternions"],
            inertial_data["velocity"],
            inertial_data["attitude"],
            target_pos=inertial_data["position"],
            target_rpy=target_rpy,  # keep current yaw,
            target_vel=target_velocity,
        )

        return rpm

    def _apply_controller_propulsion(
        self, velocity: np.ndarray, flight_state_manager: FlightStateManager
    ):
        rpm = self._apply_controller(velocity, flight_state_manager)
        self.motors.apply(rpm)

    def _apply_direct_velocity_propulsion(self, velocity: np.ndarray):
        self.direct_velocity_applier.apply(velocity)

    def propel(self, velocity: np.ndarray, flight_state_manager: FlightStateManager):
        if not self.use_direct_velocity:
            self._apply_controller_propulsion(velocity, flight_state_manager)
        else:
            self._apply_direct_velocity_propulsion(velocity)
