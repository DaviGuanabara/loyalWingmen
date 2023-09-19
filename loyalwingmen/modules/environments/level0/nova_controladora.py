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
        control_frequency: float = 240.0,
        use_quadcopter_model: bool = False,
    ):
        self.use_quadcopter_model = use_quadcopter_model
        self.dt = 1 / control_frequency  # environmental_parameters.timestep
        self.operational_constraints = operational_constraints
        self.quadcopter_specs = quadcopter_specs
        self.MAX_RPM = operational_constraints.max_rpm
        self.MIN_RPM = -self.MAX_RPM  # self.MAX_RPM / 3

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

        self.pid_for_x = PID(
            0.01,
            0.1,
            1,
            max_output_value=max_xy_thrust,
            min_output_value=-max_xy_thrust,
        )
        self.pid_for_y = PID(
            0.01,
            0.1,
            1,
            max_output_value=max_xy_thrust,
            min_output_value=-max_xy_thrust,
        )
        self.pid_for_z = PID(
            5.2 * 1e-9,
            0,
            0,
            max_output_value=max_z_thrust,
            min_output_value=-max_z_thrust,
        )

        self.pid_tor_x = PID(
            0.2,
            0.01,
            1,
            max_output_value=max_xy_torque,
            min_output_value=-max_xy_torque,
        )
        self.pid_tor_y = PID(
            -0.2,
            -0.01,
            -1,
            max_output_value=max_xy_torque,
            min_output_value=-max_xy_torque,
        )
        self.pid_tor_z = PID(
            1,
            1,
            0.1,
            max_output_value=max_z_torque,
            min_output_value=-max_z_torque,
        )

        # Inicializa a QuadcopterDynamics
        self.quadcopter_mathematical_model = QuadcopterDynamics(
            quadcopter_specs, operational_constraints
        )

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        self.MIXER_MATRIX = np.array(
            [[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]]
        )

    def reset(self):
        pass

    def get_orientation_from_vector(self, forward) -> np.ndarray:
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

        foward_intensity = np.linalg.norm(forward)
        foward_direction = forward / (foward_intensity if foward_intensity > 0 else 1)

        if np.abs(foward_direction[2]) < 0.9999:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])

        right = np.cross(foward_direction, up)
        right_intensity = np.linalg.norm(right)
        right = right / (right_intensity if right_intensity > 0 else 1)

        up = np.cross(right, foward_direction)
        rotation_matrix = np.array([right, up, foward_direction])

        rotation = Rotation.from_matrix(rotation_matrix)

        # TODO: Verificar se é necessário converter para ângulos de Euler
        # PROBLEMA DE GIMBAL LOCK
        try:
            roll, pitch, yaw = rotation.as_euler("zyx", degrees=False)
        except UserWarning:
            print("Erro ao converter quaternion para ângulos de Euler.")

            roll, yaw = 0.0, 0.0  # ou quaisquer valores padrão que você achar adequado
            pitch = 0.0
            print(pitch, roll, yaw)

        return np.array([roll, pitch, yaw])

    def _desired_forces(
        self,
        desired_velocity: np.ndarray,
        current_velocity: np.ndarray,
        current_acceleration: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        for_x = self.pid_for_x.compute(
            desired_velocity[0], current_velocity[0], current_acceleration[0], dt
        )
        for_y = self.pid_for_y.compute(
            desired_velocity[1], current_velocity[1], current_acceleration[1], dt
        )
        for_z = self.pid_for_z.compute(
            desired_velocity[2], current_velocity[2], current_acceleration[2], dt
        )
        return np.array([for_x, for_y, for_z])

    def _desired_torques(
        self,
        desired_attitude: np.ndarray,
        current_attitude: np.ndarray,
        current_angular_rate: np.ndarray,
        dt,
    ) -> np.ndarray:
        tor_x = self.pid_tor_x.compute(
            desired_attitude[0], current_attitude[0], current_angular_rate[0], dt
        )
        tor_y = self.pid_tor_y.compute(
            desired_attitude[1], current_attitude[1], current_angular_rate[1], dt
        )
        tor_z = self.pid_tor_z.compute(
            desired_attitude[2], current_attitude[2], current_angular_rate[2], dt
        )
        torques = np.array([tor_x, tor_y, tor_z])

        # This magic values were found in DSLPID CONTROLLER. So, i kept using them.
        return np.clip(torques, -3200, 3200, dtype=np.float32)

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
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(
            3, 3
        )
        scalar_thrust = max(0.0, np.dot(desired_thrust, cur_rotation[:, 2]))
        thrust_each_motor = math.sqrt(scalar_thrust / (4 * self.KF))
        rpm = thrust_each_motor + np.dot(self.MIXER_MATRIX, desired_torques)

        return np.clip(rpm, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)

    def _compute_model_compensations(
        self,
        desired_velocity,
        desired_attitude,
        calculated_rpm,
        dt,
        flight_state: Dict[str, np.ndarray],
    ):
        self.quadcopter_mathematical_model.update(flight_state)
        self.quadcopter_mathematical_model.compute_dynamics(calculated_rpm, dt)
        inertial_data = self.quadcopter_mathematical_model.get_inertial_data()

        compensation_forces = self._desired_forces(
            desired_velocity,
            inertial_data["velocity"],
            inertial_data["acceleration"],
            dt,
        )
        compensation_torques = self._desired_torques(
            desired_attitude,
            inertial_data["attitude"],
            inertial_data["angular_rate"],
            dt,
        )

        return compensation_forces, compensation_torques

    def compute_rpm(
        self,
        desired_velocity: np.ndarray,
        flight_state: Dict[str, np.ndarray],
        dt: float,
    ):
        """
        Atualiza os controladores e retorna os comandos dos motores.

        :param desired_velocity: np.ndarray, Velocidade desejada [vx, vy, vz]
        :param flight_state: Dict, Estado atual do voo
        :return: np.ndarray, Comandos dos motores (RPMs)
        """
        self.update_alpha(flight_state)

        # Get State
        # ============================================
        current_velocity = flight_state["velocity"]
        current_acceleration = flight_state["acceleration"]

        current_attitude = flight_state["attitude"]
        current_quaternion = flight_state["quaternions"]
        current_angular_rate = flight_state["angular_rate"]

        # Get Desired Forces and Torques
        # ============================================
        desired_forces = self._desired_forces(
            desired_velocity, current_velocity, current_acceleration, dt
        )
        # the desired forces shows the orientation that the quadcopter should have.
        desired_attitude = self.get_orientation_from_vector(desired_forces)
        desired_torques = self._desired_torques(
            desired_attitude, current_attitude, current_angular_rate, dt
        )

        # Compute RPMs
        # ============================================
        calculated_rpm = self.forces_torques_to_rpm(
            desired_forces, desired_torques, current_quaternion
        )

        if not self.use_quadcopter_model:
            return np.clip(calculated_rpm, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)

        # Compute compensations from math model
        # ============================================
        compensation_forces, compensation_torques = self._compute_model_compensations(
            desired_velocity, desired_attitude, calculated_rpm, dt, flight_state
        )

        final_forces = desired_forces + (self.alpha * compensation_forces)
        final_torques = desired_torques + (self.alpha * compensation_torques)

        self.last_desired_velocity = desired_velocity.copy()
        final_rpms = self.forces_torques_to_rpm(
            final_forces, final_torques, current_quaternion
        )

        return np.clip(final_rpms, self.MIN_RPM, self.MAX_RPM, dtype=np.float32)


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        max_output_value: float = 100.0,
        min_output_value: float = -100.0,
        limit_output_on=False,
        anti_windup_on=False,
        max_integral_value: float = 100.0,
        min_integral_value: float = -100.0,
        derivative_filter_on=False,
        enhance_derivative_with_estimate_on=False,
        deadband=0.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.deadband = deadband
        self.prev_error = 0.0
        self.integral = 0.0

        self.limit_output_on = limit_output_on
        self.anti_windup_on = anti_windup_on
        self.derivative_filter_on = derivative_filter_on
        self.enhance_derivative_with_estimate_on = enhance_derivative_with_estimate_on

        if limit_output_on:
            self.max_output_value = max_output_value
            self.min_output_value = min_output_value

        if anti_windup_on:
            self.max_integral_value = max_integral_value
            self.min_integral_value = min_integral_value

        if derivative_filter_on:
            self.filtered_derivative = 0.0
            self.derivative_filter_alpha = 0.9

    def _compute_derivative(self, error, current_value_rate: float, dt: float) -> float:
        raw_derivative = current_value_rate

        if self.enhance_derivative_with_estimate_on:
            raw_derivative += ((error - self.prev_error) / dt) if dt > 0 else 0.0

        if self.derivative_filter_on:
            self.filtered_derivative = (
                (1 - self.derivative_filter_alpha) * raw_derivative
                + self.derivative_filter_alpha * self.filtered_derivative
            )
            raw_derivative = self.filtered_derivative

        return raw_derivative

    def _compute_integral(self, error, dt: float) -> float:
        self.integral += error * dt
        if self.anti_windup_on:
            self.integral = np.clip(
                self.integral, self.min_integral_value, self.max_integral_value
            )

        return self.integral

    def compute(
        self,
        desired_value: float,
        current_value: float,
        current_value_rate: float,
        dt: float,
    ) -> float:
        error = desired_value - current_value
        if abs(error) < self.deadband:
            error = 0.0

        proportional = self.kp * error
        integral = self.ki * self._compute_integral(error, dt)
        derivative = self.kd * self._compute_derivative(error, current_value_rate, dt)

        self.prev_error = error
        output = proportional + integral + derivative

        if self.limit_output_on:
            return np.clip(output, self.min_output_value, self.max_output_value)

        return output

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

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
        self._initialize_inertial_data()

    def _initialize_inertial_data(self):
        self._inertial_data = {
            "position": np.zeros(3),
            "attitude": np.zeros(3),
            "velocity": np.zeros(3),
            "angular_rate": np.zeros(3),
            "quaternion": np.zeros(4),
            "acceleration": np.zeros(3),
            "angular_acceleration": np.zeros(3),
        }

    def reset(self):
        self._initialize_inertial_data()

    def update(self, inertial_data: dict):
        if hasattr(self, "_inertial_data"):
            self._inertial_data.update(inertial_data)
        else:
            self._inertial_data = inertial_data.copy()

    def get_inertial_data(self):
        return self._inertial_data.copy()

    def _compute_forces(
        self, rpm: np.ndarray, quaternion: np.ndarray, force_coefficient, weight
    ) -> Tuple[np.ndarray, np.ndarray]:
        rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        forces = np.array(rpm**2) * force_coefficient
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        return thrust_world_frame - np.array([0, 0, weight]), forces

    def _compute_torques(
        self,
        rpm: np.ndarray,
        forces: np.ndarray,
        angular_rates: np.ndarray,
        torque_coefficient,
        arm_length,
        inertia_matrix,
    ) -> np.ndarray:
        """
        arm_length = L
        torque_coefficient = KM
        inertia_matrix = J

        Gyroscopic effect of quadcopter propeler:
        'np.cross(angular_rates, np.dot(inertia_matrix, angular_rates))'
        has to be subtracted from the torques
        it is an effect that occurs when the angular velocity of the quadcopter
        is not aligned with the angular momentum of the rotors

        """
        torque_each_motor = np.array(rpm**2) * torque_coefficient
        z_torque = (
            -torque_each_motor[0]
            + torque_each_motor[1]
            - torque_each_motor[2]
            + torque_each_motor[3]
        )
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (
            arm_length / np.sqrt(2)
        )
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (
            arm_length / np.sqrt(2)
        )

        raw_torques = np.array([x_torque, y_torque, z_torque])
        return raw_torques - np.cross(
            angular_rates, np.dot(inertia_matrix, angular_rates)
        )

    def _update_state_by_cinematic(
        self, angular_rate, acceleration, angular_acceleration, dt
    ):
        position = self._inertial_data["position"]
        attitude = self._inertial_data["attitude"]
        velocity = self._inertial_data["velocity"]

        self._inertial_data["position"] = (
            position + velocity * dt + 0.5 * acceleration * dt**2
        )
        self._inertial_data["attitude"] = (
            attitude + angular_rate * dt + 0.5 * angular_acceleration * dt**2
        )
        self._inertial_data["quaternion"] = p.getQuaternionFromEuler(
            self._inertial_data["attitude"]
        )
        self._inertial_data["velocity"] = velocity + acceleration * dt
        self._inertial_data["angular_rate"] = angular_rate + angular_acceleration * dt

    def compute_dynamics(self, rpm, dt):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################

        force_coefficient = self._specifications.KF
        torque_coefficient = self._specifications.KM
        arm_length = self._specifications.L
        inertia_matrix = self._specifications.J
        inverse_inertia_matrix = self._specifications.J_INV
        weight = self._specifications.WEIGHT
        mass = self._specifications.M

        quaternion = self._inertial_data["quaternion"]
        angular_rate = self._inertial_data["angular_rate"]

        #### Compute forces and torques ############################
        force_world_frame, forces = self._compute_forces(
            rpm, quaternion, force_coefficient, weight
        )
        torques = self._compute_torques(
            rpm, forces, angular_rate, torque_coefficient, arm_length, inertia_matrix
        )

        acceleration = force_world_frame / mass
        angular_acceleration = np.dot(inverse_inertia_matrix, torques)

        #### Update state ##########################################
        self._update_state_by_cinematic(
            angular_rate, acceleration, angular_acceleration, dt
        )
