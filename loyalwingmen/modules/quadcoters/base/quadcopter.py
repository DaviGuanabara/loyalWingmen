import numpy as np
import pybullet as p

from typing import Dict
from enum import Enum, auto

from ..components.dataclasses.quadcopter_specs import QuadcopterSpecs
from ..components.dataclasses.operational_constraints import OperationalConstraints

from ..components.sensors.lidar import LiDAR
from ..components.sensors.imu import InertialMeasurementUnit

from ..components.dataclasses.flight_state import (
    FlightStateManager,
    FlightStateDataType,
)
from ..components.actuators.propulsion import PropulsionSystem

from ...environments.dataclasses.environment_parameters import EnvironmentParameters

from loyalwingmen.modules.utils.enums import DroneModel
from loyalwingmen.modules.events.message_hub import MessageHub


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
        droneSpecs: QuadcopterSpecs,
        operational_constraints: OperationalConstraints,
        environment_parameters: EnvironmentParameters,
        quadcopter_type=QuadcopterType.QUADCOPTER,
        use_direct_velocity: bool = False,
    ):
        # ... other initializations ...

        self.check_quadcopter_type()

        self.id: int = id
        self.model: DroneModel = model
        self.droneSpecs: QuadcopterSpecs = droneSpecs
        self.operational_constraints: OperationalConstraints = operational_constraints
        self.environment_parameters: EnvironmentParameters = environment_parameters
        self.quadcopter_type = quadcopter_type
        self.use_direct_velocity = use_direct_velocity

        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self._setup()

    def check_quadcopter_type(self):
        """
        Ensure that the quadcopter type is valid.

        Raises:
            ValueError: If the quadcopter type is invalid.
        """
        # Allowing for future quadcopter types besides LW and LM
        if self.quadcopter_type == QuadcopterType.QUADCOPTER:
            raise ValueError(
                f"You should choose a quadcopter type, not {self.quadcopter_type}"
            )

    def _setup(self):
        """
        Set up the initial state and components of the quadcopter.
        This includes initializing the flight state manager, sensors, message hub, and propulsion system.
        """
        self.flight_state_manager: FlightStateManager = FlightStateManager()
        self.imu = InertialMeasurementUnit(self.id, client_id=self.client_id)
        self.lidar = LiDAR(self.id, client_id=self.client_id)

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic="flight_state", subscriber=self._subscriber_flight_state
        )
        self.propulsion_system = PropulsionSystem(
            self.id,
            self.model,
            self.droneSpecs,
            self.environment_parameters,
            self.use_direct_velocity,
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

    def _compute_velocity_from_command(self, motion_command: np.ndarray) -> np.ndarray:
        """
        Compute the velocity vector from a given motion command.

        Parameters:
        - motion_command: The motion command where the first three elements represent direction,
        and the fourth element represents intensity or magnitude.

        Returns:
        - velocity: The computed velocity vector.
        """
        norm = np.linalg.norm(motion_command[:3])
        if norm == 0:
            return np.array([0, 0, 0])

        intensity = self.operational_constraints.speed_limit * motion_command[3]
        return intensity * motion_command[:3] / norm

    def drive(self, motion_command: np.ndarray):
        """
        Apply the given motion command to the quadcopter's propulsion system.
        Parameters:
        - motion_command: The command to be applied. This could be RPM values, thrust levels, etc.
        """

        velocity = self._compute_velocity_from_command(motion_command)
        self.propulsion_system.propel(velocity, self.flight_state_manager)

    # =================================================================================================================
    # Delete or Destroy
    # =================================================================================================================

    def detach_from_simulation(self):
        self.messageHub.terminate(self.id)
        p.removeBody(self.id, physicsClientId=self.client_id)
        self.id = Quadcopter.INVALID_ID
