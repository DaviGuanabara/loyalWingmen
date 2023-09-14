import pybullet as p
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict, List
from time import time

from ...quadcoters.quadcopter_factory import (
    QuadcopterFactory,
    Quadcopter,
    QuadcopterType,
    LoyalWingman,
    LoiteringMunition,
    OperationalConstraints,
    FlightStateDataType,
    LoiteringMunitionBehavior,
    CommandType,
)

from ...quadcoters.components.controllers.DSLPIDControl import DSLPIDControl

from ..helpers.normalization import normalize_flight_state

from ..dataclasses.environment_parameters import EnvironmentParameters


class PIDTuningSimulation:
    def __init__(self, environment_parameters: EnvironmentParameters):
        """
        Initializes the PID tuning simulation.

        Args:
        - environment_parameters: Configuration for the environment.
        """
        self.target_velocity = np.ones(3)
        self.drone_initial_state = np.zeros(3)
        self.command_type = CommandType.VELOCITY_TO_CONTROLLER
        self.environment_parameters = environment_parameters

        self.factory = QuadcopterFactory(environment_parameters)
        self.start_time = time()
        self.reset()

    def _init_loyalwingman(
        self, initial_position: np.ndarray, initial_angular: np.ndarray
    ):
        """
        Initializes the loyal wingman drone with given position and angular position.
        """
        self.loyalwingman = self.factory.create_loyalwingman(
            initial_position, initial_angular, self.command_type
        )
        self.loyalwingman.update_imu()

    def _compute_control(
        self, controller: DSLPIDControl, target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Computes the control RPMs using the given controller and target velocity.
        """
        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        yaw = (
            inertial_data["attitude"][2]
            if inertial_data["attitude"] is not None
            else 0.0
        )
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
            target_rpy=target_rpy,
            target_vel=target_velocity,
        )

        return rpm

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs one step in the simulation given the action.
        """
        pid_coefficientes = self._preprocesss_action(action)
        controller = DSLPIDControl(
            self.loyalwingman.model,
            self.loyalwingman.droneSpecs,
            self.environment_parameters,
            pid_coefficientes,
        )
        rpm = self._compute_control(controller, self.target_velocity)

        self.loyalwingman.drive(rpm)

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            p.stepSimulation()

        # Compute reward and check termination criteria
        self.loyalwingman.update_imu()
        terminated = self.is_done()
        reward = self.compute_reward()
        observation = self.compute_observation()

        return observation, reward, terminated, {}

    def compute_observation(self) -> np.ndarray:
        """
        Computes the observation for the agent.
        """
        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        # This is just an example, modify as per the desired observation
        return inertial_data["velocity"]

    def _preprocesss_action(self, action: List[float]) -> Dict[str, np.ndarray]:
        """
        Processes the raw action into PID coefficients.
        """
        keys = [
            "P_COEFF_FOR",
            "I_COEFF_FOR",
            "D_COEFF_FOR",
            "P_COEFF_TOR",
            "I_COEFF_TOR",
            "D_COEFF_TOR",
        ]
        assert len(action) == 3 * len(keys), "Invalid action length"
        return {
            keys[i]: np.array(action[3 * i : 3 * (i + 1)]) for i in range(len(keys))
        }

    def generate_target_velocity(self) -> np.ndarray:
        """
        Generates a random target velocity.
        """
        # Generating a 3D velocity vector
        return np.random.uniform(-1, 1, 3)

    def reset(self) -> None:
        """
        Resets the simulation environment.
        """
        initial_position = np.zeros(3)
        initial_angular_position = np.zeros(3)
        self.target_velocity = np.ones(3)
        self._init_loyalwingman(initial_position, initial_angular_position)
        self.start_time = time()

    def compute_reward(self) -> float:
        """
        Computes the reward for the current state.
        """
        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        actual_velocity = inertial_data["velocity"]
        velocity_difference = np.linalg.norm(actual_velocity - self.target_velocity)

        return float(-velocity_difference)

    def is_done(self) -> bool:
        """
        Determines if the episode should terminate.
        """
        elapsed_time = time() - self.start_time
        if elapsed_time > 20:
            return True

        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        actual_velocity = inertial_data["velocity"]
        velocity_difference = np.linalg.norm(actual_velocity - self.target_velocity)

        return bool(velocity_difference > 0.5)
