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
        self.MAX_RPM = self.MAX_RPM / 2
        self.HOVER_RPM = operational_constraints.hover_rpm
        self.M = quadcopter_specs.M
        self.L = quadcopter_specs.L
        self.WEIGHT = quadcopter_specs.WEIGHT
        self.KF = quadcopter_specs.KF
        self.KM = quadcopter_specs.KM
        self.GRAVITY_ACCELERATION = environmental_parameters.G
        self.J = quadcopter_specs.J
        self.J_INV = quadcopter_specs.J_INV

        self.pid_vx = PID(0.1, 0.1, 0.1)
        self.pid_vy = PID(0.1, 0.1, 0.1)
        self.pid_vz = PID(0.1, 0.1, 0.1)

        self.pid_roll = PID(
            0.1, 0.1, 0.1
        )  # Exemplo de valores, ajuste conforme necessário
        self.pid_pitch = PID(0.1, 0.1, 0.1)
        self.pid_yaw = PID(0.1, 0.1, 0.1)
        # Inicializa os auto-tuners
        self.tuner_vx = PIDAutoTuner(self.pid_vx)
        self.tuner_vy = PIDAutoTuner(self.pid_vy)
        self.tuner_vz = PIDAutoTuner(self.pid_vz)

        # Inicializa a QuadcopterDynamics
        self.dynamics = QuadcopterDynamics(quadcopter_specs, operational_constraints)

    def velocity_to_attitude(self, desired_velocity: np.ndarray):
        """
        Calcula a atitude desejada com base na velocidade desejada.

        :param desired_velocity: Velocidade desejada.
        :return: Atitude desejada [roll, pitch, yaw].
        """
        # Suposição simplificada: o roll e o pitch desejados são proporcionais às velocidades desejadas em x e y.
        desired_roll = desired_velocity[
            0
        ]  # isso é apenas uma suposição, ajuste conforme necessário!
        desired_pitch = desired_velocity[1]
        desired_yaw = 0.0  # supondo que o yaw desejado seja constante
        return np.array([desired_roll, desired_pitch, desired_yaw])

    def update(
        self,
        desired_velocity: np.ndarray,
        current_velocity: np.ndarray,
        current_attitude: np.ndarray,
    ):
        """
        Atualiza os controladores e retorna os comandos dos motores.

        :param desired_velocity: np.ndarray, Velocidade desejada [vx, vy, vz]
        :param current_velocity: np.ndarray, Velocidade atual [vx, vy, vz]
        :return: np.ndarray, Comandos dos motores (RPMs)
        """

        dt = self.dt

        # Calcule os erros de velocidade e obtenha a saída PID
        vx_output = self.pid_vx.compute(desired_velocity[0] - current_velocity[0], dt)
        vy_output = self.pid_vy.compute(desired_velocity[1] - current_velocity[1], dt)
        vz_output = self.pid_vz.compute(desired_velocity[2] - current_velocity[2], dt)

        # Converta a saída PID em forças desejadas (por simplicidade, assumindo que a saída PID representa uma força diretamente)
        # Note que a direção z pode necessitar de uma força adicional para combater a gravidade
        desired_forces = np.array(
            [vx_output, vy_output, vz_output + self.M * self.GRAVITY_ACCELERATION]
        )

        # Calcule os erros de atitude e obtenha a saída PID
        roll_output = self.pid_roll.compute(
            0 - current_attitude[0], dt
        )  # O valor desejado é 0
        pitch_output = self.pid_pitch.compute(0 - current_attitude[1], dt)
        yaw_output = self.pid_yaw.compute(0 - current_attitude[2], dt)

        # Converta a saída PID em torques desejados
        desired_torques = np.array([roll_output, pitch_output, yaw_output])

        return self.dynamics.forces_torques_to_rpm(desired_forces, desired_torques)


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        output_limits=(float("-inf"), float("inf")),
        anti_windup=True,
        integral_limits=(float("-inf"), float("inf")),
        deadband=0.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        self.integral_limits = integral_limits
        self.deadband = deadband

        self.prev_error = 0.0
        self.integral = 0.0

        self.derivative_filter_alpha = 0.9
        self.filtered_derivative = 0.0
        self.active = True

    def compute(self, current_value: float, dt: float) -> float:
        if not self.active:
            return 0.0

        error = self.setpoint - current_value
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
        return max(min(output, self.output_limits[1]), self.output_limits[0])

    def is_active(self) -> bool:
        return self.active

    def update_set_point(self, set_point: float):
        self.setpoint = set_point

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


class PIDAutoTuner:
    def __init__(self, pid_controller: PID):
        self.pid = pid_controller

    def reset(self):
        self.pid.reset()

    def ziegler_nichols_first_order(self, K, T):
        kp = 0.6 * (T / K)
        ti = 2 * T
        td = 0.5 * T

        ki = kp / ti
        kd = kp * td

        return kp, ki, kd

    def cohen_coon(self, K, T):
        kp = (4.0 / 3.0) * (T / K)
        ti = T * ((32.0 + 6 * np.sqrt(6)) / (13.0 + 8 * np.sqrt(6)))
        td = T * (4.0 / (11.0 + 2 * np.sqrt(6)))

        ki = kp / ti
        kd = kp * td

        return kp, ki, kd

    def auto_tune(self, system, set_point, agent_frequency, method="zn", duration=10):
        K, T = self.analyze_step_response(system, set_point, agent_frequency, duration)
        if method == "zn":
            kp, ki, kd = self.ziegler_nichols_first_order(K, T)
        elif method == "cc":
            kp, ki, kd = self.cohen_coon(K, T)
        else:
            raise ValueError(
                "Invalid tuning method. Choose either 'zn' for Ziegler-Nichols or 'cc' for Cohen-Coon."
            )

        self.pid.update_gains(kp, ki, kd)
        return kp, ki, kd

    def analyze_step_response(self, system, set_point, agent_frequency, duration=10):
        dt = 1.0 / agent_frequency
        max_time_to_settle = 1.0 / agent_frequency

        time_elapsed = 0.0
        prev_output = system(0)
        initial_output = prev_output

        while time_elapsed < duration:
            output = system(set_point)

            if output >= 0.632 * set_point:
                T = time_elapsed
                K = (output - initial_output) / (time_elapsed * set_point)
                break

            prev_output = output
            time_elapsed += dt

        else:
            raise ValueError(
                "The system did not respond as expected within the given duration."
            )

        if T > max_time_to_settle:
            raise Warning(
                f"The system took {T} seconds to reach 63.2% of the set point. It should have taken less than {max_time_to_settle} seconds."
            )

        return K, T

    def start_tuning(
        self, system, set_point, agent_frequency, method="zn", duration=10
    ):
        kp, ki, kd = self.auto_tune(
            system, set_point, agent_frequency, method, duration
        )
        print(f"Tuned Gains: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")


class QuadcopterDynamics:
    def __init__(self, specs: QuadcopterSpecs, constraints: OperationalConstraints):
        self._specifications = specs
        self._constraints = constraints
        self._initialize_inertial_state()

    def _initialize_inertial_state(self):
        self._inertial_state = {
            "position": np.zeros(3),
            "euler_angles": np.zeros(3),
            "linear_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "linear_acceleration": np.zeros(3),
            "angular_acceleration": np.zeros(3),
        }

    def _gravitational_force(self) -> np.ndarray:
        return np.array([0, 0, self._constraints.weight])

    def forces_torques_to_rpm(self, forces: np.ndarray, torques: np.ndarray):
        """
        Converte forças e torques desejados em RPMs dos motores.
        :param forces: Forças desejadas.
        :param torques: Torques desejados.
        :return: RPMs dos motores.
        """

        max_torques = np.array(
            [
                self._constraints.max_xy_torque,
                self._constraints.max_xy_torque,
                self._constraints.max_z_torque,
            ]
        )

        forces = np.clip(forces, 0, self._constraints.max_thrust)
        torques = np.clip(torques, 0, max_torques)

        F_total, tau_x, tau_y, tau_z = forces[2], torques[0], torques[1], torques[2]

        # Resolvendo o sistema de equações para RPM^2
        KF = self._specifications.KF
        KM = self._specifications.KM
        L = self._specifications.L
        RPM2_1 = (
            1
            / (4 * KF)
            * (F_total - 2 * KF / L * tau_y + 2 * KF / L * tau_x - tau_z / KM)
        )
        RPM2_2 = (
            1
            / (4 * KF)
            * (F_total - 2 * KF / L * tau_y - 2 * KF / L * tau_x + tau_z / KM)
        )
        RPM2_3 = (
            1
            / (4 * KF)
            * (F_total + 2 * KF / L * tau_y - 2 * KF / L * tau_x - tau_z / KM)
        )
        RPM2_4 = (
            1
            / (4 * KF)
            * (F_total + 2 * KF / L * tau_y + 2 * KF / L * tau_x + tau_z / KM)
        )

        # Calculando os RPMs
        RPM_1 = np.sqrt(RPM2_1)
        RPM_2 = np.sqrt(RPM2_2)
        RPM_3 = np.sqrt(RPM2_3)
        RPM_4 = np.sqrt(RPM2_4)

        # Limitando os RPMs aos valores de MIN_RPM e MAX_RPM
        RPMs = np.array([RPM_1, RPM_2, RPM_3, RPM_4])
        RPMs = np.clip(RPMs, 0, self._constraints.max_rpm)

        return RPMs

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

    def integrate_quaternion(self, q, angular_velocity, dt):
        """Integrate angular velocity to get the updated quaternion."""
        w, x, y, z = q
        p, q, r = angular_velocity

        dqdt = np.array(
            [
                0.5 * (-x * p - y * q - z * r),
                0.5 * (w * p + y * r - z * q),
                0.5 * (w * q - x * r + z * p),
                0.5 * (w * r + x * q - y * p),
            ]
        )

        q_new = q + dqdt * dt
        return q_new / np.linalg.norm(q_new)

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
        return {"velocity": state["linear_velocity"], "acceleration": acceleration}

    def angular_dynamics(self, state, body_torque):
        """Compute angular dynamics derivatives."""
        inertia_tensor = self._specifications.J
        inv_inertia_tensor = self._specifications.J_INV
        angular_drag = np.cross(
            state["angular_velocity"], np.dot(inertia_tensor, state["angular_velocity"])
        )
        angular_acceleration = np.dot(inv_inertia_tensor, body_torque - angular_drag)
        return {
            "angular_velocity": state["angular_velocity"],
            "angular_acceleration": angular_acceleration,
        }

    def rk4(self, f, state, u, dt):
        """4th order Runge-Kutta integration."""
        k1 = f(state, u)
        k2 = f({key: state[key] + 0.5 * dt * k1[key] for key in state}, u)
        k3 = f({key: state[key] + 0.5 * dt * k2[key] for key in state}, u)
        k4 = f({key: state[key] + dt * k3[key] for key in state}, u)
        return {
            key: state[key] + dt / 6 * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
            for key in state
        }

    def update_dynamics(self, rpm: np.ndarray, timestep: float) -> dict:
        thrust_vectors, body_torque = self.compute_motor_outputs(rpm)

        linear_state = {
            "linear_velocity": self._inertial_state["linear_velocity"],
            "position": self._inertial_state["position"],
        }
        updated_linear_state = self.rk4(
            self.linear_dynamics, linear_state, thrust_vectors, timestep
        )
        self._inertial_state["linear_velocity"] = updated_linear_state[
            "linear_velocity"
        ]
        self._inertial_state["position"] = updated_linear_state["position"]

        angular_state = {
            "angular_velocity": self._inertial_state["angular_velocity"],
            "euler_angles": self._inertial_state["euler_angles"],
        }
        updated_angular_state = self.rk4(
            self.angular_dynamics, angular_state, body_torque, timestep
        )
        self._inertial_state["angular_velocity"] = updated_angular_state[
            "angular_velocity"
        ]

        current_quaternion = self.euler_to_quaternion(
            *self._inertial_state["euler_angles"]
        )
        updated_quaternion = self.integrate_quaternion(
            current_quaternion, self._inertial_state["angular_velocity"], timestep
        )
        self._inertial_state["euler_angles"] = self.quaternion_to_euler(
            *updated_quaternion
        )

        return self._inertial_state.copy()
