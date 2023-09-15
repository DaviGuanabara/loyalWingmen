import numpy as np
import pybullet as p

from .actuator_interface import ActuatorInterface
from ..controllers.DSLPIDControl import DSLPIDControl

from ..dataclasses.flight_state import FlightStateManager


from ....environments.helpers.environment_parameters import EnvironmentParameters
from ..dataclasses.quadcopter_specs import QuadcopterSpecs


from ....utils.enums import DroneModel

from enum import Enum, auto


class CommandType(Enum):
    VELOCITY_TO_CONTROLLER = auto()
    VELOCITY_DIRECT = auto()
    RPM = auto()


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

        """
            This data were extracted from dls_pid_control.py.
        """
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.minimum_rpm = (self.MIN_PWM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        self.maximum_rpm = (self.MAX_PWM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

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
        drone_model: DroneModel,
        drone_specs: QuadcopterSpecs,
        environment_parameters: EnvironmentParameters,
        quadcopter_name: str = "",
        command_type: CommandType = CommandType.VELOCITY_TO_CONTROLLER,
    ):
        self.quadcopter_name = quadcopter_name
        self.environment_parameters = environment_parameters
        self.command_type = command_type

        self.propeller = self._init_propeller(
            command_type, drone_id, drone_specs, drone_model, environment_parameters
        )

    def _init_propeller(
        self, command_type, drone_id, drone_specs, drone_model, environment_parameters
    ):
        if command_type == CommandType.VELOCITY_DIRECT:
            direct_velocity_applier = DirectVelocityApplier(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            return lambda velocity, flight_state_manager: direct_velocity_applier.apply(
                velocity
            )

        elif command_type == CommandType.VELOCITY_TO_CONTROLLER:
            controller = DSLPIDControl(
                drone_model, drone_specs, environment_parameters
            )  # default or custom controller
            motors = Motors(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            return lambda target_velocity, flight_state_manager: self._velocity_to_controller(
                target_velocity, flight_state_manager, controller, motors
            )

        elif command_type == CommandType.RPM:
            motors = Motors(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            return lambda rpm, flight_state_manager: motors.apply(rpm)

        else:
            raise ValueError(f"Invalid command type: {command_type}")

        # return lambda any_thing, flight_state_manager: print("Invalid command type")

    def _velocity_to_controller(
        self, target_velocity, flight_state_manager, controller, motors
    ):
        rpms = self._apply_controller(target_velocity, flight_state_manager, controller)
        motors.apply(rpms)

    def _apply_controller(
        self,
        target_velocity: np.ndarray,
        flight_state_manager: FlightStateManager,
        controller,
    ):
        inertial_data = flight_state_manager.get_inertial_data()

        yaw = (inertial_data["attitude"] or np.zeros(3))[2]
        target_rpy = np.array([0, 0, yaw])

        aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps
        timestep_period = self.environment_parameters.timestep_period
        control_timestep = aggregate_physics_steps * timestep_period

        rpm, _, _ = controller.computeControl(
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

    def propel(self, velocity: np.ndarray, flight_state_manager: FlightStateManager):
        self.propeller(velocity, flight_state_manager)
