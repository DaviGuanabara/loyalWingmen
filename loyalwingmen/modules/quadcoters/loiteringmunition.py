import numpy as np
import pybullet as p

from .base.quadcopter import (
    Quadcopter,
    FlightStateManager,
    EnvironmentParameters,
    QuadcopterType,
    OperationalConstraints,
    QuadcopterSpecs,
    DroneModel,
)

from enum import Enum


class LoiteringMunitionBehavior(Enum):
    FROZEN = 1
    STRAIGHT_LINE = 2
    CIRCLE = 3


class LoiteringMunition(Quadcopter):
    def __init__(
        self,
        id: int,
        model: DroneModel,
        droneSpecs: QuadcopterSpecs,
        operationalConstraints: OperationalConstraints,
        environment_parameters: EnvironmentParameters,
        quadcopter_name: str,
        use_direct_velocity: bool = False,
    ):
        use_direct_velocity = True
        super().__init__(
            id,
            model,
            droneSpecs,
            operationalConstraints,
            environment_parameters,
            QuadcopterType.LOITERINGMUNITION,
            quadcopter_name,
            use_direct_velocity,
        )
        self.quadcopter_name: str = quadcopter_name

        self.behavior_map = {
            LoiteringMunitionBehavior.FROZEN: self._frozen,
            LoiteringMunitionBehavior.STRAIGHT_LINE: self._straight_line,
            LoiteringMunitionBehavior.CIRCLE: self._circle,
        }

        self.set_behavior(LoiteringMunitionBehavior.FROZEN)

    def _frozen(self, flight_state: FlightStateManager):
        return np.array([0, 0, 0, 0])

    def _straight_line(self, flight_state: FlightStateManager, direction, intensity):
        return np.array([*direction, intensity])

    def _circle(
        self,
        flight_state: FlightStateManager,
        radius=1,
        origin: np.ndarray = np.array([0, 0, 0]),
        velocity: np.ndarray = np.array([0, 0, 0]),
    ):
        timeset_period = self.environment_parameters.timestep_period
        aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps

        quad_position = flight_state.get_data("position")["position"] or np.array(
            [0, 0, 0]
        )
        quad_velocity = velocity

        # Calculate center of circle
        distance_to_origin = np.linalg.norm(quad_position)

        center_direction = (origin - 1 * quad_position) / distance_to_origin
        center = quad_position + center_direction * radius

        # Calculate Aceleration:
        quad_velocity_intensity = np.linalg.norm(quad_velocity)
        aceleration_direction = (center - quad_position) / np.linalg.norm(
            center - quad_position
        )
        aceleration_intensity = quad_velocity_intensity**2 / radius
        aceleration = aceleration_direction * aceleration_intensity

        # Calculate new velocity
        period = timeset_period * aggregate_physics_steps
        new_velocity = quad_velocity + aceleration * period

        # Calculate New Command
        new_intensity = np.linalg.norm(new_velocity)
        new_direction = new_velocity / np.linalg.norm(new_intensity)

        return np.concatenate([new_direction, new_intensity])

    def set_behavior(self, behavior: LoiteringMunitionBehavior):
        if behavior == LoiteringMunitionBehavior.FROZEN:
            self.behavior_function = lambda flight_state: self.behavior_map[behavior](
                flight_state
            )

        elif behavior == LoiteringMunitionBehavior.STRAIGHT_LINE:
            direction = np.random.rand(3)
            intensity = np.random.uniform(0.01, 0.5)
            self.behavior_function = lambda flight_state: self.behavior_map[behavior](
                flight_state, direction, intensity
            )

        elif behavior == LoiteringMunitionBehavior.CIRCLE:
            radius = 1
            origin = np.array([0, 0, 0])
            direction = np.random.rand(3)
            intensity = np.random.uniform(0.01, 0.5)
            self.behavior_function = lambda flight_state: self.behavior_map[behavior](
                flight_state, radius, origin, velocity=intensity * direction
            )

        self.current_behavior = behavior

    def drive_via_behavior(self):
        flight_state_manager = self.flight_state_manager
        command = self.behavior_function(flight_state_manager)
        print(f"loitering munition command: {command}")
        self.drive(command)
