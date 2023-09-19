import numpy as np
from enum import Enum
from typing import Dict, Optional

from ...quadcoters.components.dataclasses.operational_constraints import (
    OperationalConstraints,
)
from ...quadcoters.components.dataclasses.quadcopter_specs import QuadcopterSpecs
from ..helpers.environment_parameters import EnvironmentParameters

from typing import Dict, List, Tuple, Union, Optional

from ...quadcoters.components.base.quadcopter import DroneModel
import numpy as np
import math
from scipy.spatial.transform import Rotation
import pybullet as p


class QuadcopterController:
    def __init__(
        self,
        operational_constraints: OperationalConstraints,
        quadcopter_specs: QuadcopterSpecs,
        environmental_parameters: EnvironmentParameters,
    ):
        self.dt = environmental_parameters.timestep
        self.operational_constraints = operational_constraints
        self.quadcopter_specs = quadcopter_specs
        self.MAX_RPM = operational_constraints.max_rpm
        self.MIN_RPM = 0  # self.MAX_RPM / 3

        max_xy_thrust = operational_constraints.max_xy_thrust
        max_z_thrust = operational_constraints.max_z_thrust

        max_xy_torque = operational_constraints.max_xy_torque
        max_z_torque = operational_constraints.max_z_torque

        print("MAX_RPM", self.MAX_RPM, "MIN_RPM", self.MIN_RPM)
        self.HOVER_RPM = operational_constraints.hover_rpm
        self.M = quadcopter_specs.M
        self.L = quadcopter_specs.L
        self.WEIGHT = quadcopter_specs.WEIGHT
        self.KF = quadcopter_specs.KF
        self.KM = quadcopter_specs.KM
        self.GRAVITY_ACCELERATION = environmental_parameters.G
        self.J = quadcopter_specs.J
        self.J_INV = quadcopter_specs.J_INV

        self.alpha = 1.0
        self.alpha_max = 2.0  # Valor máximo para alpha, ajuste conforme necessário
        self.alpha_min = 0  # Valor mínimo para alpha, ajuste conforme necessário
        self.alpha_step = 0.01
        self.last_desired_velocity = np.zeros(3)  # Inicialização com vetor zero

        self.pid_for_x = PID(0.04, 0.4, 1.25, max_xy_thrust)
        self.pid_for_y = PID(0.005, 0.05, 0.05, max_xy_thrust)
        self.pid_for_z = PID(0.02, 0.2, 0.5, max_z_thrust)

        self.pid_tor_x = PID(-70000.0, 70000.0, 60000.0, max_xy_torque)
        self.pid_tor_y = PID(-70000.0, 70000.0, 60000.0, max_xy_torque)
        self.pid_tor_z = PID(-20000.0, 20000.0, 12000.0, max_z_torque)

        # Inicializa a QuadcopterDynamics
        self.dynamics = QuadcopterDynamics(quadcopter_specs, operational_constraints)

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        self.MIXER_MATRIX = np.array(
            [[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]]
        )

    def reset(self):
        pass

    def cartesian_to_euler_angles(self, forward) -> np.ndarray:
        """
        ### Função `cartesian_to_quaternions`

        Essa função tem como objetivo converter uma direção cartesiana no espaço, especificamente a direção de força desejada para um drone, em um quaternion que representa a orientação do drone.

        #### Argumentos:
        - `forward`: Um vetor numpy representando a direção de força desejada no espaço.

        #### Retorno:
        - Um objeto `Rotation` da biblioteca SciPy, que representa a rotação desejada em formato de quaternion.

        #### Descrição do Processo:

        1. **Normalização da Direção de Força**: O vetor de força desejada é tratado como uma direção "para a frente". É essencial que esse vetor esteja normalizado para garantir cálculos precisos.
        2. **Calculando o vetor 'Up'**: Um vetor "up" arbitrário é calculado com base na direção de força. Se a força não estiver quase alinhada com o eixo Z, o eixo Z é usado como vetor "up". Se estiver quase alinhado, o eixo Y é utilizado.
        3. **Calculando o vetor 'Right'**: Usando o produto vetorial entre a direção de força e o vetor "up", um vetor "right" é calculado. Este vetor é normalizado para garantir que tenha magnitude 1.
        4. **Atualização do vetor 'Up'**: Para garantir ortogonalidade, o vetor "up" é recalculado como o produto vetorial entre "right" e a direção de força.
        5. **Construção da Matriz de Rotação**: A matriz de rotação é construída usando os vetores "right", "up" e a direção de força.
        6. **Conversão para Quaternion**: A matriz de rotação é então convertida para um quaternion usando a função `from_matrix` da classe `Rotation` da biblioteca SciPy.

        """

        # 1. Trate o vetor de força desejado como uma direção de "frente".
        # (É importante que este vetor seja normalizado)
        foward_intensity = np.linalg.norm(forward)

        foward_direction = forward / (foward_intensity if foward_intensity > 0 else 1)

        # 2. Calcula um vetor "up" arbitrário.
        # Para simplificar, se o forward não é alinhado com z, podemos usar z como o "up" vector.
        # Se for alinhado, podemos usar y.
        if np.abs(foward_direction[2]) < 0.9999:  # Se não é quase alinhado com z
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])

        # 3. Obtenha o vetor direito como o produto cruzado de "forward" e "up"
        right = np.cross(foward_direction, up)
        right_intensity = np.linalg.norm(right)
        right = right / (right_intensity if right_intensity > 0 else 1)

        # Atualiza o vetor "up" para garantir que seja perfeitamente perpendicular a "forward"
        up = np.cross(right, foward_direction)

        # 4. Construa a matriz de rotação usando os vetores direito, up e forward
        rotation_matrix = np.array([right, up, foward_direction])

        # 5. Converta a matriz de rotação para um quaternion
        rotation_quaternion = Rotation.from_matrix(rotation_matrix)
        return self.safe_as_euler(rotation_quaternion)

    def safe_as_euler(self, rotation: Rotation) -> np.ndarray:
        roll, pitch, yaw = rotation.as_euler("zyx", degrees=False)

        # Ajustando o valor limite para radianos, próximo de +/- π/2 (ou +/- 90 graus)
        if np.abs(pitch) > (
            np.pi / 2 - 0.05
        ):  # Estamos nos aproximando de um gimbal lock
            roll = 0.0

        return np.array([roll, pitch, yaw])

    def _desired_forces(
        self, desired_velocity: np.ndarray, current_velocity
    ) -> np.ndarray:
        for_x = self.pid_for_x.compute(
            desired_velocity[0], current_velocity[0], self.dt
        )
        for_y = self.pid_for_y.compute(
            desired_velocity[1], current_velocity[1], self.dt
        )
        for_z = self.pid_for_z.compute(
            desired_velocity[2], current_velocity[2], self.dt
        )
        return np.array([for_x, for_y, for_z])

    def _desired_torques(
        self, desired_attitude: np.ndarray, current_attitude
    ) -> np.ndarray:
        tor_x = self.pid_tor_x.compute(
            desired_attitude[0], current_attitude[0], self.dt
        )
        tor_y = self.pid_tor_y.compute(
            desired_attitude[1], current_attitude[1], self.dt
        )
        tor_z = self.pid_tor_z.compute(
            desired_attitude[2], current_attitude[2], self.dt
        )
        return np.array([tor_x, tor_y, tor_z])

    def update_alpha(self, flight_state: Dict[str, np.ndarray]):
        if not hasattr(self, "last_desired_velocity"):
            return

        current_velocity = flight_state["velocity"]
        last_disered_velocity = self.last_desired_velocity
        error = np.linalg.norm(current_velocity - last_disered_velocity)

        alpha = self.alpha
        alpha += float(0.01 * error)
        self.alpha = max(self.alpha_min, min(alpha, self.alpha_max))

    def forces_torques_to_rpm(
        self,
        desired_thrust: np.ndarray,
        desired_torques: np.ndarray,
        current_quaternion: np.ndarray,
    ):
        # Combine forces and torques

        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(
            3, 3
        )
        scalar_thrust = max(0.0, np.dot(desired_thrust, cur_rotation[:, 2]))
        thrust_each_motor = math.sqrt(scalar_thrust / (4 * self.KF))
        rpm = thrust_each_motor + np.dot(self.MIXER_MATRIX, desired_torques)

        return np.clip(rpm, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)

    def compute_rpm(
        self,
        desired_velocity: np.ndarray,
        flight_state: Dict[str, np.ndarray],
    ):
        """
        Atualiza os controladores e retorna os comandos dos motores.

        :param desired_velocity: np.ndarray, Velocidade desejada [vx, vy, vz]
        :param flight_state: Dict, Estado atual do voo
        :return: np.ndarray, Comandos dos motores (RPMs)
        """
        self.update_alpha(flight_state)

        dt = self.dt
        # desired_attitude = flight_state[]#np.zeros(3)

        current_velocity = flight_state["velocity"]
        current_attitude = flight_state["attitude"]
        current_quaternion = flight_state["quaternions"]

        desired_forces = self._desired_forces(desired_velocity, current_velocity)
        # the desired forces shows the orientation that the quadcopter should have.
        desired_attitude = self.cartesian_to_euler_angles(desired_forces)
        desired_torques = self._desired_torques(desired_attitude, current_attitude)
        desired_rpm = self.forces_torques_to_rpm(
            desired_forces, desired_torques, current_quaternion
        )

        self.dynamics.reset(flight_state)
        simulated_final_state = self.dynamics.compute_dynamics(desired_rpm, dt)

        simulated_final_velocity = simulated_final_state["velocity"]
        simulated_final_attitude = simulated_final_state["attitude"]

        compensation_forces = self._desired_forces(
            desired_velocity, simulated_final_velocity
        )
        compensation_torques = self._desired_torques(
            desired_attitude, simulated_final_attitude
        )

        final_forces = desired_forces + (self.alpha * compensation_forces)
        final_torques = desired_torques + (self.alpha * compensation_torques)

        self.last_desired_velocity = desired_velocity.copy()
        final_rpms = self.forces_torques_to_rpm(
            final_forces, final_torques, current_quaternion
        )

        # return np.clip(final_rpms, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)
        return np.clip(desired_rpm, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_output_value: float = 0.0,
        min_output_value: float = 0.0,
        deadband=0.0,
        anti_windup=True,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd

        self.max_output_value = max_output_value
        self.min_output_value = min_output_value

        self.anti_windup = anti_windup

        self.integral_limits = (
            (-10, 10) if anti_windup else (float("-inf"), float("inf"))
        )
        self.deadband = deadband

        self.prev_error = 0.0
        self.integral = 0.0

        self.derivative_filter_alpha = 0.9
        self.filtered_derivative = 0.0
        self.active = True

    def compute(self, desired_value: float, current_value: float, dt: float) -> float:
        if not self.active:
            return 0.0

        error = desired_value - current_value
        if abs(error) < self.deadband:
            error = 0.0

        proportional = self.kp * error

        self.integral += error * dt
        if self.anti_windup:
            self.integral = max(
                min(self.integral, self.integral_limits[1]), self.integral_limits[0]
            )
        integral_value = self.ki * self.integral

        raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.filtered_derivative = (
            1 - self.derivative_filter_alpha
        ) * raw_derivative + self.derivative_filter_alpha * self.filtered_derivative
        derivative_value = self.kd * self.filtered_derivative

        self.prev_error = error

        output = proportional + integral_value + derivative_value

        return max(min(output, self.max_output_value), self.min_output_value)

    def is_active(self) -> bool:
        return self.active

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def toggle(self):
        self.active = not self.active

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0

    def update_gains(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd


class QuadcopterDynamics:
    def __init__(self, specs: QuadcopterSpecs, constraints: OperationalConstraints):
        self._specifications = specs
        self._constraints = constraints
        self._initialize_inertial_state()

    def _initialize_inertial_state(self):
        self._inertial_state = {
            "position": np.zeros(3),
            "attitude": np.zeros(3),
            "velocity": np.zeros(3),
            "angular_rate": np.zeros(3),
            "acceleration": np.zeros(3),
            "angular_acceleration": np.zeros(3),
        }

    def reset(self, flight_state: Optional[dict] = None):
        if flight_state is None:
            self._initialize_inertial_state()
        else:
            # TODO: Flight_state vai acabar adicionando mais coisas no dicionário
            # Não acredito que isso vá ser um problema
            # para o bem da simplicidade, vou manter assim.
            self._inertial_state.update(flight_state)

    def _gravitational_force(self) -> np.ndarray:
        return np.array([0, 0, self._constraints.weight])

    def compute_motor_outputs(self, rpm: np.ndarray) -> tuple:
        """Compute thrust and body torque based on motor RPM."""
        motor_thrusts = rpm**2 * self._specifications.KF
        body_torque = rpm**2 * self._specifications.KM
        body_torque_sum = np.sum(
            np.array([(-1) ** i * torque for i, torque in enumerate(body_torque)])
        )
        thrust_vectors = [np.array([0, 0, thrust]) for thrust in motor_thrusts]
        body_torque_vector = np.array([0, 0, body_torque_sum])
        return thrust_vectors, body_torque_vector

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to a quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return np.array([w, x, y, z])

    def quaternion_to_euler(self, w, x, y, z):
        """Convert a quaternion to Euler angles."""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = min(t2, +1.0)
        t2 = max(t2, -1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw])

    def integrate_quaternion(self, q, angular_rate, dt):
        """Integrate angular rate to get the updated quaternion."""
        w, x, y, z = q
        p, q, r = angular_rate

        dqdt = np.array(
            [
                0.5 * (-x * p - y * q - z * r),
                0.5 * (w * p + y * r - z * q),
                0.5 * (w * q - x * r + z * p),
                0.5 * (w * r + x * q - y * p),
            ]
        )

        q_array = np.array([w, x, y, z])
        q_new = q_array + dqdt * dt
        norm = np.linalg.norm(q_new)
        return q_new / norm if norm > 0 else np.array([1, 0, 0, 0])

    def linear_dynamics(self, state, thrust_vectors):
        """Compute linear dynamics derivatives."""
        total_thrust = np.sum(thrust_vectors, axis=0)
        gravitational_force = np.array([0, 0, self._constraints.weight])
        acceleration = (total_thrust - gravitational_force) / self._specifications.M
        acceleration = np.clip(
            acceleration,
            -self._constraints.acceleration_limit,
            self._constraints.acceleration_limit,
        )
        return {"velocity": state["velocity"], "acceleration": acceleration}

    def angular_dynamics(self, state, body_torque):
        """Compute angular dynamics derivatives."""
        inertia_tensor = self._specifications.J
        inv_inertia_tensor = self._specifications.J_INV
        angular_drag = np.cross(
            state["angular_rate"], np.dot(inertia_tensor, state["angular_rate"])
        )
        angular_acceleration = np.dot(inv_inertia_tensor, body_torque - angular_drag)
        return {
            "angular_rate": state["angular_rate"],
            "angular_acceleration": angular_acceleration,
        }

    def rk4(self, f, state, u, dt):
        """4th order Runge-Kutta integration."""

        k1 = f(state, u)
        k2 = f({key: state[key] + 0.5 * dt * k1.get(key, 0) for key in state}, u)
        k3 = f({key: state[key] + 0.5 * dt * k2.get(key, 0) for key in state}, u)
        k4 = f({key: state[key] + dt * k3.get(key, 0) for key in state}, u)
        return {
            key: state[key]
            + dt
            / 6
            * (
                k1.get(key, 0)
                + 2 * k2.get(key, 0)
                + 2 * k3.get(key, 0)
                + k4.get(key, 0)
            )
            for key in state
        }

    def euler_integration(self, f, state, u, dt):
        """Euler integration."""

        derivative = f(state, u)
        return {key: state[key] + dt * derivative.get(key, 0) for key in state}

    def compute_dynamics(self, rpm: np.ndarray, timestep: float) -> dict:
        thrust_vectors, body_torque = self.compute_motor_outputs(rpm)

        print("thrust_vectors", thrust_vectors, "body_torque", body_torque)
        linear_state = {
            "velocity": self._inertial_state["velocity"],
            "position": self._inertial_state["position"],
        }

        print("linear_state", linear_state)
        updated_linear_state = self.euler_integration(
            self.linear_dynamics, linear_state, thrust_vectors, timestep
        )

        print("updated_linear_state", updated_linear_state)
        self._inertial_state["velocity"] = updated_linear_state["velocity"]
        self._inertial_state["position"] = updated_linear_state["position"]

        angular_state = {
            "angular_rate": self._inertial_state["angular_rate"],
            "attitude": self._inertial_state["attitude"],
        }
        print("angular_state", angular_state)
        updated_angular_state = self.rk4(
            self.angular_dynamics, angular_state, body_torque, timestep
        )
        self._inertial_state["angular_rate"] = updated_angular_state["angular_rate"]

        print("_inertial_state", self._inertial_state)

        current_quaternion = self.euler_to_quaternion(*self._inertial_state["attitude"])
        updated_quaternion = self.integrate_quaternion(
            current_quaternion, self._inertial_state["angular_rate"], timestep
        )
        self._inertial_state["attitude"] = self.quaternion_to_euler(*updated_quaternion)

        return self._inertial_state.copy()
