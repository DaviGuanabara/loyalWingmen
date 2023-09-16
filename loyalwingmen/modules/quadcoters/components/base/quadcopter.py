import numpy as np
import pybullet as p

from typing import Dict
from enum import Enum, auto

from ....environments.helpers.environment_parameters import EnvironmentParameters

from ....utils.enums import DroneModel
from ....events.message_hub import MessageHub

from ..dataclasses.quadcopter_specs import QuadcopterSpecs
from ..dataclasses.operational_constraints import OperationalConstraints

from ..sensors.lidar import LiDAR
from ..sensors.imu import InertialMeasurementUnit

from ..dataclasses.flight_state import (
    FlightStateManager,
    FlightStateDataType,
)
from ..actuators.propulsion import PropulsionSystem, CommandType


class QuadcopterType(Enum):
    QUADCOPTER = auto()
    LOYALWINGMAN = auto()
    LOITERINGMUNITION = auto()


class Quadcopter:
    INVALID_ID = -1

    def __init__(
        self,
        id: int,
        model: DroneModel,
        quadcopter_specs: QuadcopterSpecs,
        operational_constraints: OperationalConstraints,
        environment_parameters: EnvironmentParameters,
        quadcopter_type=QuadcopterType.QUADCOPTER,
        quadcopter_name: str = "",
        command_type: CommandType = CommandType.VELOCITY_DIRECT,
    ):
        self.check_quadcopter_type(quadcopter_type)

        self.id: int = id
        self.model: DroneModel = model
        self.quadcopter_specs: QuadcopterSpecs = quadcopter_specs
        self.operational_constraints: OperationalConstraints = operational_constraints
        self.environment_parameters: EnvironmentParameters = environment_parameters
        self.quadcopter_type = quadcopter_type
        self.quadcopter_name = quadcopter_name

        self.command_type = command_type

        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self._setup()

    def check_quadcopter_type(self, quadcopter_type):
        """
        Ensure that the quadcopter type is valid.

        Raises:
            ValueError: If the quadcopter type is invalid.
        """
        # Allowing for future quadcopter types besides LW and LM
        if quadcopter_type == QuadcopterType.QUADCOPTER:
            raise ValueError(
                f"You should choose a quadcopter type, not {self.quadcopter_type}"
            )

    def _setup(self):
        """
        Set up the initial state and components of the quadcopter.
        This includes initializing the flight state manager, sensors, message hub, and propulsion system.
        """
        self.flight_state_manager: FlightStateManager = FlightStateManager()
        self.imu = InertialMeasurementUnit(
            self.id, self.client_id, self.environment_parameters
        )
        self.lidar = LiDAR(self.id, client_id=self.client_id)

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic="flight_state", subscriber=self._subscriber_flight_state
        )
        self.propulsion_system = PropulsionSystem(
            self.id,
            self.model,
            self.quadcopter_specs,
            self.environment_parameters,
            self.operational_constraints,
            self.quadcopter_name,
            self.command_type,
        )

    # =================================================================================================================

    def replace(self, position: np.ndarray, quaternion: np.ndarray):
        p.resetBasePositionAndOrientation(self.id, position, quaternion)
        self.update_imu()

    # =================================================================================================================
    # Sensors
    # =================================================================================================================

    def update_imu(self):
        self.imu.update_data()
        imu_data = self.imu.read_data()
        self._update_flight_state(imu_data)
        self._publish_flight_state()

    def update_lidar(self):
        self.lidar.update_data()
        lidar_data = self.lidar.read_data()
        self._update_flight_state(lidar_data)
        # Note: We don't publish the flight state here

    def get_lidar_shape(self) -> tuple:
        return self.lidar.get_data_shape()

    def reset_sensors(self):
        self.imu.reset()
        self.lidar.reset()

    # =================================================================================================================
    # Flight State
    # =================================================================================================================

    def _update_flight_state(self, sensor_data: Dict):
        self.flight_state_manager.update_data(sensor_data)

    @property
    def flight_state(self) -> Dict:
        return self.flight_state_manager.get_data() or {}

    def flight_state_by_type(self, flight_state_data_type: FlightStateDataType) -> Dict:
        return self.flight_state_manager.get_data_by_type(flight_state_data_type) or {}

    def reset_flight_state(self):
        self.flight_state_manager = FlightStateManager()

    # =================================================================================================================
    # Communication
    # =================================================================================================================

    def _publish_flight_state(self):
        """
        Publish the current flight state data to the message hub.
        """
        inertial_data = self.flight_state_manager.get_inertial_data()

        message = {**inertial_data, "publisher_type": self.quadcopter_type}

        self.messageHub.publish(
            topic="flight_state", message=message, publisher_id=self.id
        )

    def _subscriber_flight_state(self, flight_state: Dict, publisher_id: int):
        """
        Handle incoming flight state data from the message hub.

        Parameters:
        - flight_state (Dict): The flight state data received.
        - publisher_id (int): The ID of the publisher of the data.
        """
        self.lidar.buffer_publisher_flight_state(flight_state, publisher_id)

    # =================================================================================================================
    # Actuators
    # =================================================================================================================

    def drive(self, motion_command: np.ndarray):
        """
        Apply the given motion command to the quadcopter's propulsion system.
        Parameters:
        - motion_command: The command to be applied. This could be RPM values, thrust levels, etc.
        """

        self.propulsion_system.propel(motion_command, self.flight_state_manager)

    # =================================================================================================================
    # Delete or Destroy
    # =================================================================================================================

    def detach_from_simulation(self):
        self.messageHub.terminate(self.id)
        p.removeBody(self.id, physicsClientId=self.client_id)
        self.id = Quadcopter.INVALID_ID
