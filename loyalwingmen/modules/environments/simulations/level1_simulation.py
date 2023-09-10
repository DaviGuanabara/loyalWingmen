import pybullet as p
import pybullet_data
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict

from ...quadcoters.quadcopter_factory import (
    QuadcopterFactory,
    Quadcopter,
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
        self,
        dome_radius: float,
        environment_parameters: EnvironmentParameters,
    ):
        self.dome_radius = dome_radius

        self.environment_parameters = environment_parameters
        self.init_simulation()

    def init_simulation(self):
        """Initialize the simulation and entities."""
        # Initialize pybullet, load plane, gravity, etc.

        if self.environment_parameters.GUI:
            client_id = self.setup_pybulley_GUI()

        else:
            client_id = self.setup_pybullet_DIRECT()

        print("Setting gravity")
        p.setGravity(
            0,
            0,
            0,  # -self.environment_parameters.G,
            physicsClientId=client_id,
        )

        print(" setRealTimeSimulation")
        p.setRealTimeSimulation(0, physicsClientId=client_id)  # No Realtime Sync

        print("setting time step")
        p.setTimeStep(
            self.environment_parameters.timestep_period,
            physicsClientId=client_id,
        )

        print("setting additional search path")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )

        print("ending resetSimulation")

        self.environment_parameters.client_id = client_id
        self.factory = QuadcopterFactory(self.environment_parameters)

        position, ang_position = self.gen_initial_position()

        print("creating loyal wingman")
        self.loyal_wingman: LoyalWingman = self.factory.create_loyalwingman(
            position, np.zeros(3), quadcopter_name="agent"
        )
        print("creating loitering munition")

        self.loitering_munition: LoiteringMunition = (
            self.factory.create_loiteringmunition(
                np.zeros(3), np.zeros(3), quadcopter_name="target"
            )
        )

        self.last_action: np.ndarray = np.zeros(4)

        self.reset()

    def setup_pybullet_DIRECT(self):
        return p.connect(p.DIRECT)

    def setup_pybulley_GUI(self):
        client_id = p.connect(p.GUI)
        for i in [
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(i, 0, physicsClientId=client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=-30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=client_id,
        )
        ret = p.getDebugVisualizerCamera(physicsClientId=client_id)

        return client_id

    def _reset_simulation(self):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Set PyBullet's parameters #############################
        print(
            "Calling p.resetSimulation(), client_id",
            self.environment_parameters.client_id,
        )
        p.resetSimulation(physicsClientId=self.environment_parameters.client_id)

    def gen_initial_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random position within the dome."""

        print("Generating initial position")
        position = np.random.rand(3)  # * (self.dome_radius / 2)
        print("position", position)
        ang_position = np.random.uniform(0, 2 * np.pi, 3)
        print("ang_position", ang_position)

        return position, ang_position

    def _housekeeping(self):
        print("Housekeeping")
        pos, ang_pos = self.gen_initial_position()
        print("pos, ang_pos", pos, ang_pos)

        target_pos, target_ang_pos = np.array([0, 0, 0]), np.array([0, 0, 0])

        if self.loyal_wingman is not None:
            print("nothing")
            self.loyal_wingman.detach_from_simulation()

        if self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyal_wingman = self.factory.create_loyalwingman(
            position=pos,
            ang_position=np.zeros(3),
            use_direct_velocity=True,
            quadcopter_name="agent",
        )
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=target_pos, ang_position=target_ang_pos, quadcopter_name="target"
        )

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)
        self.start_time = time()

    def reset(self):
        """Reset the simulation to its initial state."""

        # self._reset_simulation()

        self._housekeeping()

        observation = self.compute_observation()
        info = self.compute_info()

        return observation, info

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation."""

        lw = self.loyal_wingman
        lm = self.loitering_munition
        # assert lw is not None, "Loyal wingman is not initialized"
        # assert lm is not None

        lw_state = lw.flight_state_by_type(FlightStateDataType.INERTIAL)
        lm_state = lm.flight_state_by_type(FlightStateDataType.INERTIAL)

        distance_vector = self._calculate_distance_vector(lw_state, lm_state)

        distance_to_target = np.linalg.norm(distance_vector)
        direction_to_target = distance_vector / max(float(distance_to_target), 1)

        distance_to_target_normalized = distance_to_target / (2 * self.dome_radius)
        lw_inertial_data_normalized = normalize_flight_state(
            lw_state, lw.operational_constraints, self.dome_radius
        )
        lm_inertial_data_normalized = normalize_flight_state(
            lm_state, lm.operational_constraints, self.dome_radius
        )

        lw_values = self.convert_dict_to_array(lw_inertial_data_normalized)
        lm_values = self.convert_dict_to_array(lm_inertial_data_normalized)

        return np.array(
            [
                *lw_values,
                *lm_values,
                *direction_to_target,
                distance_to_target_normalized,
                *self.last_action,
            ],
            dtype=np.float32,
        )

    def distance(self, lw_state: Dict, lm_state: Dict) -> float:
        distance_vector = self._calculate_distance_vector(lw_state, lm_state)
        return float(np.linalg.norm(distance_vector))

    def observation_size(self):
        position = 3
        velocity = 3
        attitude = 3
        angular_rate = 3

        lw_inertial_data_shape = position + velocity + attitude + angular_rate
        lm_inertial_data_shape = position + velocity + attitude + angular_rate

        direction_to_target_shape = 3
        distance_to_target_shape = 1
        last_action_shape = 4

        return (
            lw_inertial_data_shape
            + lm_inertial_data_shape
            + direction_to_target_shape
            + distance_to_target_shape
            + last_action_shape
        )

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        end_time = time()
        elapsed_time = end_time - self.start_time
        max_episode_time = self.environment_parameters.max_episode_time

        if max_episode_time > 0 and elapsed_time > max_episode_time:
            return True

        if self.is_outside_dome(self.loitering_munition):
            return True

        if self.is_outside_dome(self.loyal_wingman):
            return True

        return False

    def step(self, rl_action: np.ndarray):
        """Execute a step in the simulation based on the RL action."""

        # assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        # assert self.loitering_munition is not None

        self.last_action = rl_action
        self.loyal_wingman.drive(rl_action)
        self.loitering_munition.drive_via_behavior()

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            p.stepSimulation()

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        self.loyal_wingman.update_lidar()
        self.loitering_munition.update_lidar()

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

        # return np.zeros(self.observation_size()), 0, False, False, {}

    def compute_truncation(self):
        return False

    def compute_info(self):
        return {}

    def compute_reward(self):
        # assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        # assert self.loitering_munition is not None

        bonus = 0
        penalty = 0
        score = 0

        lw_flight_state = self.loyal_wingman.flight_state
        lm_flight_state = self.loitering_munition.flight_state

        distance_vector = self._calculate_distance_vector(
            lw_flight_state, lm_flight_state
        )
        distance = np.linalg.norm(distance_vector)

        score = self.dome_radius - distance

        if distance < 0.2:
            bonus += 1_000

        if distance > self.dome_radius:
            penalty += 1_000

        return score + bonus - penalty

    def _calculate_distance_vector(self, lw_state: Dict, lm_state: Dict) -> np.ndarray:
        lw_pos = lw_state.get("position", np.zeros(3))
        lm_pos = lm_state.get("position", np.zeros(3))
        return lm_pos - lw_pos

    def is_outside_dome(self, entity: Quadcopter) -> bool:
        """Check if the entity is outside the dome radius."""
        inertial_data = entity.flight_state_by_type(FlightStateDataType.INERTIAL)
        position = inertial_data.get("position", np.zeros(3))

        return float(np.linalg.norm(position)) > self.dome_radius

    def replace_loitering_munition(self):
        """Reset the loitering munition to a random position within the dome."""

        position, ang_position = self.gen_initial_position()

        assert self.loitering_munition is not None
        self.loitering_munition.detach_from_simulation()
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=position, ang_position=ang_position
        )

        self.loitering_munition.update_imu()

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate((array, value))

        return array

    def close(self):
        p.disconnect(physicsClientId=self.environment_parameters.client_id)
