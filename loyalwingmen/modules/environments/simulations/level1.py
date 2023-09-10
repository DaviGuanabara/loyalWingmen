import pybullet as p
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict

from ...quadcoters.quadcopter_factory import (
    QuadcopterFactory,
    DroneType,
    LoyalWingman,
    LoiteringMunition,
    OperationalConstraints,
    FlightStateDataType,
    LoiteringMunitionBehavior,
)

from ..helpers.normalization import normalize_flight_state

from ..dataclasses.environment_parameters import EnvironmentParameters
from time import time


class DroneChaseStaticTargetSimulation:
    def __init__(
        self, dome_radius: float, environment_parameters: EnvironmentParameters
    ):
        self.dome_radius = dome_radius

        self.environment_parameters = environment_parameters
        self.init_simulation()

    def init_simulation(self):
        """Initialize the simulation and entities."""
        # Initialize pybullet, load plane, gravity, etc.
        # TODO: Add pybullet initialization here

        self.factory = QuadcopterFactory(self.environment_parameters)
        self.loyal_wingman: Optional[LoyalWingman] = None
        self.loitering_munition: Optional[LoiteringMunition] = None
        self.last_action: np.ndarray = np.zeros(4)

        self.reset()

    def gen_initial_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random position within the dome."""

        position = np.random.rand(3) * self.dome_radius
        ang_position = np.random.uniform(0, 2 * np.pi, 3)

        return position, ang_position

    def reset(self):
        """Reset the simulation to its initial state."""

        pos, ang_pos = self.gen_initial_position()
        target_pos, target_ang_pos = np.array([0, 0, 0]), np.array([0, 0, 0])

        if self.loyal_wingman is not None:
            self.loyal_wingman.detach_from_simulation()

        if self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyal_wingman = self.factory.create_loyalwingman(
            position=pos, ang_position=ang_pos
        )
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=target_pos, ang_position=target_ang_pos
        )

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)

        self.start_time = time()

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation."""

        lw = self.loyal_wingman
        lm = self.loitering_munition
        assert lw is not None, "Loyal wingman is not initialized"
        assert lm is not None

        lw_state = lw.flight_state_by_type(FlightStateDataType.INERTIAL)
        lm_state = lm.flight_state_by_type(FlightStateDataType.INERTIAL)

        distance_vector = self._extracted_from_compute_reward_12(lw_state, lm_state)
        distance_to_target = np.linalg.norm(distance_vector)
        direction_to_target = distance_vector / max(float(distance_to_target), 1e-9)

        lw_norm = normalize_flight_state(
            lw_state, lw.operational_constraints, self.dome_radius
        )
        lm_norm = normalize_flight_state(
            lm_state, lm.operational_constraints, self.dome_radius
        )

        lw_values = self.convert_dict_to_array(lw_norm)
        lm_values = self.convert_dict_to_array(lm_norm)

        return np.array(
            [
                *lw_values,
                *lm_values,
                *direction_to_target,
                distance_to_target,
                *self.last_action,
            ],
            dtype=np.float32,
        )

    def distance(self, lw_state: Dict, lm_state: Dict) -> float:
        distance_vector = self._extracted_from_compute_reward_12(lw_state, lm_state)
        return float(np.linalg.norm(distance_vector))

    def observation_shape(self):
        lw_array_shape = 3
        lm_array_shape = 3
        last_action_shape = 4
        direction_to_target_shape = 3
        distance_to_target_shape = 1

        return (
            lw_array_shape
            + lm_array_shape
            + last_action_shape
            + direction_to_target_shape
            + distance_to_target_shape
        )

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        end_time = time()
        elapsed_time = end_time - self.start_time
        max_episode_time = self.environment_parameters.max_episode_time

        if max_episode_time > 0 and elapsed_time > max_episode_time:
            return True

        assert self.loyal_wingman is not None

        return bool(self.is_outside_dome(self.loyal_wingman))

    def step(self, rl_action: np.ndarray):
        """Execute a step in the simulation based on the RL action."""
        # Convert RL action to quadcopter commands
        # TODO: Convert action to quadcopter commands

        # Check distances and positions
        assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        assert self.loitering_munition is not None

        self.last_action = rl_action
        self.loyal_wingman.drive(rl_action)
        self.loitering_munition.execute_behavior()

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            p.stepSimulation()

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        # self.loyalwingman.update_lidar()
        # self.loitering_munition.update_lidar()

        lw_inertial_data = self.loyal_wingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        lm_inertial_data = self.loitering_munition.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        if self.distance(lw_inertial_data, lm_inertial_data) < 0.2:
            self.replace_loitering_munition()

        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = self.compute_termination()

        truncated = self.compute_truncation()
        info = self.compute_info()

        return observation, reward, terminated, truncated, info

    def compute_truncation(self):
        return False

    def compute_info(self):
        return {}

    def compute_reward(self):
        assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        assert self.loitering_munition is not None

        bonus = 0
        penalty = 0
        score = 0

        lw_flight_state = self.loyal_wingman.flight_state
        lm_flight_state = self.loitering_munition.flight_state

        distance_vector = self._extracted_from_compute_reward_12(
            lw_flight_state, lm_flight_state
        )
        distance = np.linalg.norm(distance_vector)

        score = self.dome_radius - distance

        if distance < 0.2:
            bonus += 1_000

        if distance > self.dome_radius:
            penalty += 1_000

        return score + bonus - penalty

    # TODO Rename this here and in `compute_observation`, `distance` and `compute_reward`
    def _extracted_from_compute_reward_12(self, arg0, arg1):
        lw_pos = arg0["position"] or np.zeros(3)
        lm_pos = arg1["position"] or np.zeros(3)
        return lm_pos - lw_pos

    def is_outside_dome(self, entity: LoyalWingman) -> bool:
        """Check if the entity is outside the dome radius."""
        inertial_data = entity.flight_state_by_type(FlightStateDataType.INERTIAL)
        position = inertial_data["position"] or np.zeros(3)

        return float(np.linalg.norm(position)) > self.dome_radius

    def replace_loitering_munition(self):
        """Reset the loitering munition to a random position within the dome."""
        random_position = np.random.uniform(-self.dome_radius, self.dome_radius, 3)
        ang_position = np.zeros(3)

        assert self.loitering_munition is not None
        self.loitering_munition.detach_from_simulation()
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=random_position, ang_position=ang_position
        )

    def reset_simulation(self):
        """Reset the entire simulation to its initial state."""
        # TODO: Reset the entities to their original positions in pybullet
        pass

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        # pairs in the dictionary
        result = dictionary.items()

        # Convert object to a list
        data = list(result)

        return np.array(data)
