from typing import NamedTuple

# Imutável
class EnvironmentParameters(NamedTuple):
    G: float
    NEIGHBOURHOOD_RADIUS: float
    simulation_frequency: int
    rl_frequency: int
    timestep_period: float
    aggregate_physics_steps: int
    client_id: int
    max_distance: float
    error: float