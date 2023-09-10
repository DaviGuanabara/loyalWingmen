import pybullet as p
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict

from ..quadcoters.quadcopter_factory import (
    QuadcopterFactory,
    DroneType,
    LoyalWingman,
    LoiteringMunition,
)
from ..environments.dataclasses.environment_parameters import EnvironmentParameters


class Simulation:
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
        self.loyalwingman: Optional[LoyalWingman] = None
        self.loitering_munition: Optional[LoiteringMunition] = None

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

        if self.loyalwingman is not None:
            self.loyalwingman.detach_from_simulation()

        if self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyalwingman = self.factory.create_loyalwingman(
            position=pos, ang_position=ang_pos
        )
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=target_pos, ang_position=target_ang_pos
        )

    def observation(self) -> Dict:
        """Return the observation of the simulation."""
        assert self.loyalwingman is not None, "Loyal wingman is not initialized"
        return self.loyalwingman.flight_state

    def calculate_done(self) -> bool:
        """Calculate if the simulation is done."""

        return bool(self.is_outside_dome(self.loyalwingman))

    def step(self, rl_action: np.ndarray):
        """Execute a step in the simulation based on the RL action."""
        # Convert RL action to quadcopter commands
        # TODO: Convert action to quadcopter commands

        # Check distances and positions
        assert self.loyalwingman is not None, "Loyal wingman is not initialized"
        assert (
            self.loitering_munition is not None
        ), "Loitering munition is not initialized"

        if self.distance(self.loyalwingman, self.loitering_munition) < 0.2:
            self.reset_loitering_munition()

        if self.is_outside_dome(self.loyalwingman):
            self.reset_simulation()

        # TODO: Execute a simulation step in pybullet

    def distance(self, lw: LoyalWingman, lm: LoiteringMunition):
        """Calculate the distance between two entities."""

        position = lw.flight_state["position"]
        target_position = lm.flight_state["position"]

        return np.linalg.norm(position - target_position)

    def is_outside_dome(self, entity):
        """Check if the entity is outside the dome radius."""
        # TODO: Check if entity's position is outside the dome
        return False

    def reset_loitering_munition(self):
        """Reset the loitering munition to a random position within the dome."""
        random_position = [
            random.uniform(-self.dome_radius, self.dome_radius),
            random.uniform(-self.dome_radius, self.dome_radius),
            random.uniform(0, self.dome_radius),  # Assuming z >= 0
        ]
        # TODO: Set the loitering munition's position to random_position in pybullet

    def reset_simulation(self):
        """Reset the entire simulation to its initial state."""
        # TODO: Reset the entities to their original positions in pybullet
        pass
