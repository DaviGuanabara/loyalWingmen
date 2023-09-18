import numpy as np
from enum import Enum
from typing import Dict, Optional

class DroneModel(Enum):
    CF2X = 1
    CF2P = 2



class QuadcopterController:
    def __init__(self, params, drone_model: DroneModel, pid_coefficients: Optional[Dict[str, np.ndarray]] = None):
        # Inicializando PIDs para controle de atitude e posição
        self.pid_roll = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        self.pid_pitch = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        self.pid_yaw = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)

        #self.pid_x = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        #self.pid_y = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        #self.pid_z = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        
        self.pid_vx = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        self.pid_vy = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)
        self.pid_vz = PID(kp=1.0, ki=0.1, kd=0.01, limit=10)

        self.MAX_RPM = params["MAX_RPM"]
        self.HOVER_RPM = params["HOVER_RPM"]
        self.MAX_SPEED_KMH = params["MAX_SPEED_KMH"]
        self.M = params["M"]
        self.L = params["L"]
        self.WEIGHT = params["WEIGHT"]
        self.KF = params["KF"]
        self.KM = params["KM"]
        self.GRAVITY_ACCELERATION = params["GRAVITY_ACCELERATION"]
        self.J = params["J"]
        self.J_INV = params["J_INV"]

        # Setup PID controllers for force and torque
        self.force_pid = {axis: PID(*params[f"force_pid_{axis}"]) for axis in ["x", "y", "z"]}
        self.torque_pid = {axis: PID(*params[f"torque_pid_{axis}"]) for axis in ["x", "y", "z"]}

        self.DRONE_MODEL = drone_model
        self.load_pid_coefficients(pid_coefficients)

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([[0.5, -0.5, -1], [0.5, 0.5, 1], [-0.5, 0.5, -1], [-0.5, -0.5, 1]])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([[0, -1, -1], [1, 0, 1], [0, 1, -1], [-1, 0, 1]])

    def load_pid_coefficients(self, pid_coefficients: Optional[Dict[str, np.ndarray]]):
        if pid_coefficients is None:
            self._default_coefficients()
        else:
            self.P_COEFF_FOR = pid_coefficients["P_COEFF_FOR"]
            self.I_COEFF_FOR = pid_coefficients["I_COEFF_FOR"]
            self.D_COEFF_FOR = pid_coefficients["D_COEFF_FOR"]
            self.P_COEFF_TOR = pid_coefficients["P_COEFF_TOR"]
            self.I_COEFF_TOR = pid_coefficients["I_COEFF_TOR"]
            self.D_COEFF_TOR = pid_coefficients["D_COEFF_TOR"]

    def _default_coefficients(self):
        self.P_COEFF_FOR = np.array([0.4, 0.4, 1.25])
        self.I_COEFF_FOR = np.array([0.05, 0.05, 0.05])
        self.D_COEFF_FOR = np.array([0.2, 0.2, 0.5])
        self.P_COEFF_TOR = np.array([70000.0, 70000.0, 60000.0])
        self.I_COEFF_TOR = np.array([0.0, 0.0, 500.0])
        self.D_COEFF_TOR = np.array([20000.0, 20000.0, 12000.0])

    def compute_motor_commands(self, desired_state, current_state):
        force_commands = {}
        torque_commands = {}

        for axis in ["x", "y", "z"]:
            # Compute desired force and torque
            force_error = desired_state["force"][axis] - current_state["force"][axis]
            torque_error = desired_state["torque"][axis] - current_state["torque"][axis]
            
            # Compute force and torque PID outputs
            force_commands[axis] = self.force_pid[axis].compute(force_error, current_state["force_derivative"][axis])
            torque_commands[axis] = self.torque_pid[axis].compute(torque_error, current_state["torque_derivative"][axis])

        # Combine forces and torques to compute motor commands. 
        # (This is a simplified representation and actual motor command computation might be more involved.)
        motor_commands = self._map_forces_torques_to_motors(force_commands, torque_commands)

        return motor_commands

    def _map_forces_torques_to_motors(self, force_commands, torque_commands):
        # Simplified force and torque to motor command mapping
        motor_outputs = {
            "motor1": force_commands["z"] + torque_commands["x"] + torque_commands["y"],
            "motor2": force_commands["z"] - torque_commands["x"] + torque_commands["y"],
            "motor3": force_commands["z"] + torque_commands["x"] - torque_commands["y"],
            "motor4": force_commands["z"] - torque_commands["x"] - torque_commands["y"],
        }
        return motor_outputs

    def control_attitude(self, desired_orientation, imu_data):
        # imu_data contém [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        roll_error = desired_orientation[0] - imu_data[0]
        pitch_error = desired_orientation[1] - imu_data[1]
        yaw_error = desired_orientation[2] - imu_data[2]

        roll_output = self.pid_roll.compute(roll_error)
        pitch_output = self.pid_pitch.compute(pitch_error)
        yaw_output = self.pid_yaw.compute(yaw_error)

        return roll_output, pitch_output, yaw_output

    def control_position_velocity(self, desired_position, desired_velocity, imu_data):
        # imu_data contém [x, y, z, vx, vy, vz]
        x_error = desired_position[0] - imu_data[0]
        y_error = desired_position[1] - imu_data[1]
        z_error = desired_position[2] - imu_data[2]

        vx_error = desired_velocity[0] - imu_data[3]
        vy_error = desired_velocity[1] - imu_data[4]
        vz_error = desired_velocity[2] - imu_data[5]

        x_output = self.pid_x.compute(x_error)
        y_output = self.pid_y.compute(y_error)
        z_output = self.pid_z.compute(z_error)

        vx_output = self.pid_vx.compute(vx_error)
        vy_output = self.pid_vy.compute(vy_error)
        vz_output = self.pid_vz.compute(vz_error)

        return x_output, y_output, z_output, vx_output, vy_output, vz_output

    def update(self, desired_orientation, desired_position, desired_velocity, imu_data):
        # Controle de atitude e posição/velocidade
        attitude_outputs = self.control_attitude(desired_orientation, imu_data[:6])
        position_velocity_outputs = self.control_position_velocity(desired_position, desired_velocity, imu_data[6:])

        # Aqui, em um cenário real, você enviaria os outputs para os motores.
        # Este código é apenas uma representação simplificada.
        
        return attitude_outputs, position_velocity_outputs

    def online_tuning(self, performance_metric):
        # Este é um exemplo simplificado. Na prática, você pode ter um algoritmo mais complexo.
        if performance_metric > 0.1:
            self.set_pid_constants('roll', 1.2, 0.1, 0.02)
        elif performance_metric < 0.05:
            self.set_pid_constants('roll', 0.8, 0.08, 0.01)
        # ... (Similar para outros eixos) ...

    def update(self, desired_orientation, desired_position, desired_velocity, imu_data, performance_metric):
        # Controle de atitude e posição/velocidade
        attitude_outputs = self.control_attitude(desired_orientation, imu_data[:6])
        position_velocity_outputs = self.control_position_velocity(desired_position, desired_velocity, imu_data[6:])
        
        # Simulando o modelo dinâmico para obter respostas dos motores
        dynamics = QuadcopterDynamics(mass=1.0)
        total_thrust = sum(position_velocity_outputs[:3])  # Exemplo simplificado
        motor_outputs = dynamics.motor_thrusts_from_forces(total_thrust, attitude_outputs)

        # Ajuste online com base no desempenho
        self.online_tuning(performance_metric)

        return motor_outputs

    def load_pid_coefficients(self, pid_coefficients: Optional[Dict[str, np.ndarray]]):
        if pid_coefficients is None:
            self._default_coefficients()
        else:
            self.P_COEFF_FOR = pid_coefficients["P_COEFF_FOR"]
            self.I_COEFF_FOR = pid_coefficients["I_COEFF_FOR"]
            self.D_COEFF_FOR = pid_coefficients["D_COEFF_FOR"]
            self.P_COEFF_TOR = pid_coefficients["P_COEFF_TOR"]
            self.I_COEFF_TOR = pid_coefficients["I_COEFF_TOR"]
            self.D_COEFF_TOR = pid_coefficients["D_COEFF_TOR"]

    def _default_coefficients(self):
        self.P_COEFF_FOR = np.array([0.4, 0.4, 1.25])
        self.I_COEFF_FOR = np.array([0.05, 0.05, 0.05])
        self.D_COEFF_FOR = np.array([0.2, 0.2, 0.5])
        self.P_COEFF_TOR = np.array([70000.0, 70000.0, 60000.0])
        self.I_COEFF_TOR = np.array([0.0, 0.0, 500.0])
        self.D_COEFF_TOR = np.array([20000.0, 20000.0, 12000.0])
    
    def reset(self):
        # Implemente a lógica de reinicialização se necessário
        pass

    def compute_motor_response(self, desired_forces, desired_torques):
        # Esta é apenas uma simulação simples para calcular a resposta do motor
        # A lógica exata dependerá do seu modelo dinâmico e dos detalhes do controlador
        
        # Calcular PWM para força e torque
        pwm_forces = self.P_COEFF_FOR * desired_forces
        pwm_torques = self.P_COEFF_TOR * desired_torques
        
        # Converta para RPM
        rpm_forces = self.PWM2RPM_SCALE * pwm_forces + self.PWM2RPM_CONST
        rpm_torques = self.PWM2RPM_SCALE * pwm_torques + self.PWM2RPM_CONST
        
        # Usar a matriz do misturador para combinar as respostas
        combined_response = np.dot(self.MIXER_MATRIX, np.hstack([rpm_forces, rpm_torques]))
        
        # Sature as respostas dentro dos limites de PWM
        return np.clip(combined_response, self.MIN_PWM, self.MAX_PWM)



class QuadcopterDynamics:
    def __init__(self, mass, gravity=9.81):
        self.mass = mass
        self.gravity = gravity

    def motor_thrusts_from_forces(self, rpm:np.ndarray, KF, KM):
        # Este é um modelo simplificado.
        # Supõe-se que os motores estão dispostos em uma configuração padrão "+".
        # torques é [tau_roll, tau_pitch, tau_yaw]
        # Retorna os empuxos para os 4 motores [m1, m2, m3, m4]

        
        torques = np.array(rpm**2) * KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        z_forces = np.array(rpm**2) * KF

        m0 =  np.array([0, 0, z_forces[0]])
        m1 =  np.array([0, 0, z_forces[1]])
        m2 =  np.array([0, 0, z_forces[2]])
        m3 =  np.array([0, 0, z_forces[3]])

        body_torque = np.array([0, 0, z_torque])
        #m1 = 0.25 * (total_thrust - torques[0] + torques[1] - torques[2])
        #m2 = 0.25 * (total_thrust + torques[0] + torques[1] + torques[2])
        #m3 = 0.25 * (total_thrust + torques[0] - torques[1] - torques[2])
        #m4 = 0.25 * (total_thrust - torques[0] - torques[1] + torques[2])
        return np.ndarray([m0, m1, m2, m3]), body_torque


class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, set_point=0.0, output_limits=(None, None), 
                 anti_windup=False, integral_limits=(None, None), deadband=0.0, 
                 proportional_on_measurement=False, controller_rate=240, agent_rate=30, disable_derivative_filter=False):
        """Inicializa o controlador PID."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        self.integral_limits = integral_limits
        self.deadband = deadband
        self.proportional_on_measurement = proportional_on_measurement
        self.active = True

        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.error_log = []

        self.derivative_filter_alpha = 1.0 if disable_derivative_filter else 0.9
        self.filtered_derivative = 0.0 

        self.controller_rate = controller_rate
        self.agent_rate = agent_rate
        self.calls_between_setpoints = self.controller_rate // self.agent_rate
        self.calls_since_last_setpoint = 0

    def compute(self, current_value, dt):
        """Calcula o valor de saída do controlador PID."""
        if not self.active:
            return 0.0
        
        # Verificação de dt
        if dt <= 0:
            return 0.0  # ou talvez retornar a última saída

        error = self.set_point - current_value

        # Deadband
        if abs(error) <= self.deadband:
            error = 0.0
        
        # Proportional on Measurement
        proportional = -current_value if self.proportional_on_measurement else error

        # Cálculo do derivativo usando o filtro passa-baixa
        raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.filtered_derivative = (1 - self.derivative_filter_alpha) * raw_derivative + self.derivative_filter_alpha * self.filtered_derivative
        self.derivative = self.filtered_derivative

        output = self.kp * proportional + self.ki * self.integral + self.kd * self.derivative
        
        # Saturação da saída
        output = self._clamp(output, *self.output_limits)

        # Acumulação do termo integral
        if not self.anti_windup or (self.output_limits[0] is None or output > self.output_limits[0]) and (self.output_limits[1] is None or output < self.output_limits[1]):
            self.integral += error * dt
            # Saturação do termo integral
            self.integral = self._clamp(self.integral, *self.integral_limits)
        
        self.prev_error = error

        # Log de erros
        self.error_log.append(error)
        if len(self.error_log) > 10:  # Manter os últimos 10 erros
            self.error_log.pop(0)

        self.calls_since_last_setpoint += 1
        if self.calls_since_last_setpoint >= self.calls_between_setpoints:
            # Resetar ou fazer algo específico, se necessário
            self.calls_since_last_setpoint = 0

        # Log de saídas
        if not hasattr(self, 'output_log'):
            self.output_log = []
        self.output_log.append(output)
        if len(self.output_log) > 10:  # Manter as últimas 10 saídas
            self.output_log.pop(0)    

        return output
    
    def is_active(self):
        """Verifica se o controlador está ativo."""
        return self.active
    
    def update_set_point(self, set_point: float):
        """Atualiza o valor de referência (set point) do controlador."""
        if set_point != self.set_point:
            self.set_point = set_point
            self.calls_since_last_setpoint = 0
            # Resetar o termo integral e o erro anterior se o setpoint mudar
            self.integral = 0.0
            self.prev_error = 0.0
        else:
            # Aqui, o setpoint não mudou. Se necessário, pode-se adicionar lógica para tratar esse caso.
            pass


    @classmethod
    def from_config(cls, config):
        """Inicializa a partir de um dicionário de configuração."""
        return cls(**config)

    def toggle(self):
        """Ativa ou desativa o controlador."""
        self.active = not self.active

    def _clamp(self, value, value_min, value_max):
        """Restringe o valor entre value_min e value_max."""
        if value_min is not None:
            value = max(value, value_min)
        if value_max is not None:
            value = min(value, value_max)
        return value

    def reset(self):
        """Reinicia os estados internos do controlador PID."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        # Resetar o registro de saída também
        if hasattr(self, 'output_log'):
            self.output_log.clear()


    def update_gains(self, kp=None, ki=None, kd=None):
        """Atualiza os ganhos do controlador PID."""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd


