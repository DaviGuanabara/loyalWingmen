import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field
from gymnasium import spaces


@dataclass(order=True)
class Kinematics:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternions: np.ndarray = field(default_factory=lambda: np.zeros(4))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def observation_space(self):
        # Define os limites para as coordenadas de posição (x, y, z)
        position_high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        position_low = -position_high

        # Define os limites para os ângulos de rotação (pitch, yaw, roll)
        angular_position_high = np.array([np.pi, np.pi, np.pi], dtype=np.float32)
        angular_position_low = -angular_position_high

        # Define os limites para os quaterniões (componentes x, y, z, w)
        quaternions_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        quaternions_low = -quaternions_high

        # Define os limites para as velocidades lineares (vx, vy, vz)
        velocity_high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        velocity_low = -velocity_high

        # Define os limites para as velocidades angulares (wx, wy, wz)
        angular_velocity_high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        angular_velocity_low = -angular_velocity_high

        # Cria o espaço de observação contendo os limites para cada componente
        return spaces.Box(low=np.concatenate([position_low, angular_position_low, quaternions_low, velocity_low, angular_velocity_low]),
                            high=np.concatenate([position_high, angular_position_high, quaternions_high, velocity_high, angular_velocity_high]),
                            dtype=np.float32)

    def observation(self):
        # Retorna um único array contendo todos os valores das variáveis do Kinematics
        return np.concatenate([self.position, self.angular_position, self.quaternions, self.velocity, self.angular_velocity], dtype=np.float32)


@dataclass(order=True)
class Parameters:
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: np.ndarray
    J_INV: np.ndarray
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: np.ndarray
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float


@dataclass(order=True)
class Informations:
    speed_limit: float = 0
    gravity: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_z_torque: float = 0
    hover_rpm: float = 0
    speed_limit: float = 0
    gnd_eff_h_clip: float = 0
    max_xy_torque: float = 0


@dataclass
class EnvironmentParameters():
    G: float
    NEIGHBOURHOOD_RADIUS: float
    simulation_frequency: int
    rl_frequency: int
    timestep_period: float
    aggregate_physics_steps: int
    client_id: int
    max_distance: float
    error: float
    debug: bool = False
