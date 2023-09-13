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


"""

Is only the agnets quadcopter that is trying to replicate the target velocity vector.
The observation is the target velocity vector, ant the action is the PID constants,
three dimensions values that will fufill the controller parameters. The reward is the
inverse of the difference between the target velocity vector and the quadcopter velocity vector.

the agent`s quadcopter starts at the origim, the target_velocty is applied to the quadcopter after the controller settings.
The controller settings comes from the action chosen by the agent.

The agent is trying to minimize the difference between the target velocity vector and the quadcopter velocity vector.
"""

class PIDTuningSimulation():
    def __init__(self, target_velocity, drone_initial_state=None):
        super(PIDTuningSimulation, self).__init__()
        
        self.target_velocity = np.array(target_velocity)
        self.drone_initial_state = drone_initial_state if drone_initial_state else self.default_initial_state()

    def default_initial_state(self):
        # Define your default initial state here
        return {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            # ... add more states if needed
        }

    def reset(self):
        self.current_state = self.drone_initial_state.copy()
        return self.current_state

    def step(self, action):
        pid_values = action  # P, I, D values from the action
        
        # Use the PID values to compute control command
        control_command = self.compute_control(pid_values)
        
        # Apply the control command to update the drone's state
        self.update_drone_state(control_command)
        
        # Compute reward
        reward = self.compute_reward()
        
        # Check termination criteria (for instance, if drone crashes or flies too high)
        done = self.is_done()
        
        return self.current_state, reward, done, {}

    def compute_control(self, pid_values):
        # Implement your PID control logic here
        # Return the control command to be applied to the drone
        pass

    def update_drone_state(self, control_command):
        # Update the drone's state based on the control command
        # This will typically involve physics computations or can be a simple model
        pass

    def compute_reward(self):
        velocity_difference = np.linalg.norm(self.current_state["velocity"] - self.target_velocity)
        return -velocity_difference  # Negative because we want to minimize this difference

    def is_done(self):
        # Define your termination criteria
        # Example: if the drone's altitude is too high or too low, terminate the episode
        return False  # Placeholder

    # ... add other necessary functions and methods
