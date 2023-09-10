from typing import Dict
import numpy as np
from ...quadcoters.components.dataclasses.operational_constraints import (
    OperationalConstraints,
)
from ..dataclasses.environment_parameters import EnvironmentParameters


def normalize_flight_state(
    flight_state: Dict,
    operational_constraints: OperationalConstraints,
    dome_radius: float,
) -> Dict:
    """Normalize the flight state of the loyal wingman."""
    position = flight_state["position"] or np.zeros(3)
    velocity = flight_state["velocity"] or np.zeros(3)
    attitude = flight_state["attitude"] or np.zeros(3)
    angular_rate = flight_state["angular_rate"] or np.zeros(3)

    norm_position = normalize_position(position, dome_radius)
    norm_velocity = normalize_velocity(velocity, operational_constraints.speed_limit)
    norm_attitude = normalize_attitude(attitude)
    norm_angular_rate = normalize_angular_rate(angular_rate)

    return {
        "position": norm_position,
        "velocity": norm_velocity,
        "attitude": norm_attitude,
        "angular_rate": norm_angular_rate,
    }


def normalize_position(position: np.ndarray, dome_radius: float):
    position /= dome_radius
    return position


def normalize_velocity(velocity: np.ndarray, max_velocity: float):
    velocity /= max_velocity
    return velocity


def normalize_attitude(attitude: np.ndarray):
    attitude /= np.pi
    return attitude


def normalize_angular_rate(angular_rate: np.ndarray):
    """Normalize the angular rate of the loyal wingman.
    Essa normalização eu retirei do projeto gym_pybullet_drones
    Não gosto dela, pois ela está retornando um vetor unitário, somente a direção do vetor angular rate
    e não a intensidade do vetor angular rate

    Normalização Gym Pybullet Drones: angular_rate/np.linalg.norm(angular_rate) if np.linalg.norm(angular_rate) != 0 else angular_rate
    Minha outra alternativa é usar um clip, com a velocidade máxima de rotação do drone sendo np.pi, mas isso não deixaria de ser uma suposição

    Ficou decidido usar o clip, para max_rate = 2*np.pi (ou seja, 1 revolução por segundo)
    """
    MAX_RATE = 2 * np.pi
    angular_rate /= MAX_RATE

    return np.clip(angular_rate, -MAX_RATE, MAX_RATE)
