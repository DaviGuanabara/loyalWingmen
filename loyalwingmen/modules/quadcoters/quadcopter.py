import numpy as np
import pybullet as p

from typing import Dict

from .components.controllers.DSLPIDControl import DSLPIDControl

from .components.dataclasses.quadcopter_specs import QuadcopterSpecs
from .components.dataclasses.operational_constraints import OperationalConstraints

from .components.sensors.sensor_interface import Sensor
from .components.sensors.imu import InertialMeasurementUnit
from .components.dataclasses.flight_state import FlightState

from ..environments.dataclasses.environment_parameters import EnvironmentParameters

from dataclasses import dataclass, field
from modules.utils.enums import DroneModel
from typing import List, TYPE_CHECKING, Union, Optional

from .components.actuators.actuator_interface import ActuatorInterface
from .components.actuators.propulsion import Propulsion, Motors, DirectVelocityApplier

from typing import Type


# TODO: CORRIGIR LIDAR, PQ LÁ NÃO TEM READ_DATA, E O READ_DATA TEM QUE RETORNAR UM DICIONÁRIO
class Quadcopter:
    def __init__(
        self,
        id: int,
        model: DroneModel,
        droneSpecs: QuadcopterSpecs,
        operationalConstraints: OperationalConstraints,
        environment_parameters: EnvironmentParameters,
        controller: Optional[DSLPIDControl] = None,
        propulsion: Optional[Propulsion] = None,
    ):
        self.id: int = id
        self.model: DroneModel = model
        self.droneSpecs: QuadcopterSpecs = droneSpecs
        self.operationalConstraints: OperationalConstraints = operationalConstraints

        self.environment_parameters: EnvironmentParameters = environment_parameters

        self.controller: Optional[DSLPIDControl] = (
            None if propulsion is None else controller
        )
        self.propulsion: Propulsion = (
            DirectVelocityApplier(
                drone_id=id,
                drone_specs=droneSpecs,
                client_id=environment_parameters.client_id,
            )
            if propulsion is None
            else propulsion
        )

        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self.flightState: FlightState = FlightState()
        self.update_flight_state()

        self.sensors: Dict[str, Sensor] = {}
        self.actuators: Dict[str, ActuatorInterface] = {}
        self.controllers: Dict[str, DSLPIDControl] = {}

    # =================================================================================================================

    def replace(self, position: np.ndarray, quaternion: np.ndarray):
        p.resetBasePositionAndOrientation(self.id, position, quaternion)
        self.update_flight_state()

    # =================================================================================================================
    # Sensors
    # =================================================================================================================

    def register_sensor(self, sensor_instance: Sensor):
        self.sensors[sensor_instance.__class__.__name__] = sensor_instance

    def unregister_sensor(self, sensor_class: Type[Sensor]):
        sensor_name = sensor_class.__name__
        if sensor_name in self.sensors:
            del self.sensors[sensor_name]

    def update_flight_state(self):
        for sensor_name, sensor in self.sensors.items():
            sensor.update_data()
            data = sensor.read_data()

            if data is not None:
                self.flightState.update_data(data)

    def get_flight_state(self) -> Dict:
        return self.flightState.get_data() or {}

    def reset_sensors(self):
        for sensor_name in self.sensors.keys():
            self.sensors[sensor_name].reset()

        self.update_flight_state()

    # =================================================================================================================
    # Actuators
    # =================================================================================================================
    def apply_propulsion(self, velocity: np.ndarray):
        """
        Apply the given command to the quadcopter's propulsion system.

        Parameters:
        - command: The command to be applied. This could be RPM values, thrust levels, etc.
        """
        try:
            assert self.controller is not None, "Controller not initialized"
            assert self.flightState is not None, "Flight state not initialized yet"
            assert self.propulsion is not None, "Propulsion not initialized"

            collected_data = {}
            necessary_keys = ["position", "quaternions", "velocity", "attitude"]
            for key in necessary_keys:
                data = self.flightState.get_data(key)
                assert data is not None, f"{key} data not found in flight state"

                collected_data[key] = data

            control_timestep = (
                self.environment_parameters.aggregate_physics_steps
                * self.environment_parameters.timestep_period
            )

            yaw = collected_data["attitude"][2]
            target_rpy = np.array([0, 0, yaw])
            rpm, _, _ = self.controller.computeControl(
                control_timestep,
                collected_data["position"],
                collected_data["quaternions"],
                collected_data["velocity"],
                collected_data["attitude"],
                target_pos=collected_data["position"],
                target_rpy=target_rpy,  # keep current yaw,
                target_vel=velocity,
            )

            self.propulsion.apply(rpm)

        except AssertionError as e:
            self.propulsion.apply(velocity)

    # =================================================================================================================
    # Flight State
    # =================================================================================================================

    def reset_flight_state(self):
        if self.flightState is not None:
            self.flightState = FlightState()
