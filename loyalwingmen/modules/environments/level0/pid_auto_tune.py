


from ...quadcoters.components.dataclasses.operational_constraints import OperationalConstraints
from ...quadcoters.components.dataclasses.quadcopter_specs import QuadcopterSpecs
from ..helpers.environment_parameters import EnvironmentParameters

from typing import Dict, List, Tuple, Union, Optional

from ...quadcoters.components.base.quadcopter import DroneModel


class PIDAutoTuner:
    def __init__(self, pid_controller):
        self.pid = pid_controller

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
            raise ValueError("Invalid tuning method. Choose either 'zn' for Ziegler-Nichols or 'cc' for Cohen-Coon.")
        
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
            raise ValueError("The system did not respond as expected within the given duration.")

        if T > max_time_to_settle:
            raise Warning(f"The system took {T} seconds to reach 63.2% of the set point. It should have taken less than {max_time_to_settle} seconds.")

        return K, T
    
    def start_tuning(self, system, set_point, agent_frequency, method="zn", duration=10):
        kp, ki, kd = self.auto_tune(system, set_point, agent_frequency, method, duration)
        print(f"Tuned Gains: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")

    def mysystem(self):
        pass
        
import numpy as np

class QuadcopterDynamics:
    def __init__(self, specs: QuadcopterSpecs, constraints: OperationalConstraints):
        self._specifications = specs
        self._constraints = constraints
        self._initialize_inertial_state()

    def _initialize_inertial_state(self):
        self._inertial_state = {
            'position': np.zeros(3),
            'euler_angles': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'linear_acceleration': np.zeros(3),
            'angular_acceleration': np.zeros(3)
        }

    def _gravitational_force(self) -> np.ndarray:
        return np.array([0, 0, self._constraints.weight])

    def compute_motor_outputs(self, rpm: np.ndarray) -> tuple:
        """Compute thrust and body torque based on motor RPM."""
        motor_thrusts = rpm**2 * self._specifications.KF 
        body_torque = rpm**2 * self._specifications.KM
        body_torque_sum = np.sum(np.array([(-1)**i * torque for i, torque in enumerate(body_torque)]))
        thrust_vectors = [np.array([0, 0, thrust]) for thrust in motor_thrusts]
        body_torque_vector = np.array([0, 0, body_torque_sum])
        return thrust_vectors, body_torque_vector
    
    def update_dynamics(self, rpm: np.ndarray, timestep: float) -> dict:
        thrust_vectors, body_torque = self.compute_motor_outputs(rpm)

        # Acceleration using Newton's second law
        total_thrust = np.sum(thrust_vectors, axis=0)
        gravitational_force = np.array([0, 0, self._constraints.weight])
        linear_acceleration = (total_thrust - gravitational_force) / self._specifications.M
        # Limit linear acceleration
        linear_acceleration = np.clip(linear_acceleration, -self._constraints.acceleration_limit, self._constraints.acceleration_limit)

        # Angular acceleration using Newton's second law for rotations
        inertia_tensor = self._specifications.J
        inv_inertia_tensor = self._specifications.J_INV
        angular_drag = np.cross(self._inertial_state['angular_velocity'], np.dot(inertia_tensor, self._inertial_state['angular_velocity']))
        angular_acceleration = np.dot(inv_inertia_tensor, body_torque - angular_drag)

        # Update velocity and position using the Euler method
        self._inertial_state['linear_velocity'] += linear_acceleration * timestep
        # Limit linear velocity
        self._inertial_state['linear_velocity'] = np.clip(self._inertial_state['linear_velocity'], -self._constraints.speed_limit, self._constraints.speed_limit)
        self._inertial_state['position'] += self._inertial_state['linear_velocity'] * timestep

        # Update angular velocity and euler angles
        self._inertial_state['angular_velocity'] += angular_acceleration * timestep
        # Limit angular velocity
        self._inertial_state['angular_velocity'] = np.clip(self._inertial_state['angular_velocity'], -self._constraints.angular_speed_limit, self._constraints.angular_speed_limit)
        self._inertial_state['euler_angles'] += self._inertial_state['angular_velocity'] * timestep  # Simplification; in reality, attitude update is more complex

        return self._inertial_state.copy()


class OperationalConstraintsCalculator:
    @staticmethod
    def compute_max_torque(max_rpm, L, KF, KM):
        max_z_torque = 2 * KM * max_rpm**2
        max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(2)

        return max_xy_torque, max_z_torque

    @staticmethod
    def compute_thrust(max_rpm, KF, M):
        max_thrust = 4 * KF * max_rpm**2
        acceleration_limit = max_thrust / M
        return max_thrust, acceleration_limit

    @staticmethod
    def compute_rpm(weight, KF, THRUST2WEIGHT_RATIO):
        max_rpm = np.sqrt((THRUST2WEIGHT_RATIO * weight) / (4 * KF))
        hover_rpm = np.sqrt(weight / (4 * KF))

        return max_rpm, hover_rpm

    @staticmethod
    def compute_speed_limit(parameters: QuadcopterSpecs):
        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1

        return VELOCITY_LIMITER * parameters.MAX_SPEED_KMH * KMH_TO_MS

    @staticmethod
    def compute_gnd_eff_h_clip(max_rpm, KF, GND_EFF_COEFF, max_thrust, PROP_RADIUS):
        return (
            0.25
            * PROP_RADIUS
            * np.sqrt((15 * max_rpm**2 * KF * GND_EFF_COEFF) / max_thrust)
        )

    @staticmethod
    def compute_moment_of_inertia(M, L):
        # I know that this may be not the correct way to compute the accelerations and velocity limits, but it is the best I can do for now.
        I_x = I_y = (1 / 12) * M * L**2
        I_z = (1 / 6) * M * L**2

        return I_x, I_y, I_z

    @staticmethod
    def compute_angular_acceleration_limit(
        max_xy_torque, max_z_torque, I_x, I_z
    ) -> float:
        alpha_x = alpha_y = max_xy_torque / I_x
        alpha_z = max_z_torque / I_z
        return max(alpha_x, alpha_z)

    @staticmethod
    def compute_angular_speed_limit(angular_acceleration_limit, timestep) -> float:
        return angular_acceleration_limit * timestep

    @staticmethod
    def compute(
        parameters: QuadcopterSpecs, environment_parameters: EnvironmentParameters
    ) -> OperationalConstraints:
        # Your operational constraints logic here
        gravity_acceleration = environment_parameters.G
        timestep = environment_parameters.timestep

        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1

        L = parameters.L
        M = parameters.M
        KF = parameters.KF
        KM = parameters.KM
        PROP_RADIUS = parameters.PROP_RADIUS
        GND_EFF_COEFF = parameters.GND_EFF_COEFF
        THRUST2WEIGHT_RATIO = parameters.THRUST2WEIGHT_RATIO

        WEIGHT = gravity_acceleration * M

        max_rpm, hover_rpm = OperationalConstraintsCalculator.compute_rpm(
            WEIGHT, KF, THRUST2WEIGHT_RATIO
        )

        speed_limit = OperationalConstraintsCalculator.compute_speed_limit(parameters)
        (
            max_thrust,
            acceleration_limit,
        ) = OperationalConstraintsCalculator.compute_thrust(max_rpm, KF, M)
        (
            max_xy_torque,
            max_z_torque,
        ) = OperationalConstraintsCalculator.compute_max_torque(max_rpm, L, KF, KM)

        gnd_eff_h_clip = OperationalConstraintsCalculator.compute_gnd_eff_h_clip(
            max_rpm, KF, GND_EFF_COEFF, max_thrust, PROP_RADIUS
        )

        I_x, I_y, I_z = OperationalConstraintsCalculator.compute_moment_of_inertia(M, L)
        angular_acceleration_limit = (
            OperationalConstraintsCalculator.compute_angular_acceleration_limit(
                max_xy_torque, max_z_torque, I_x, I_z
            )
        )
        angular_speed_limit = (
            OperationalConstraintsCalculator.compute_angular_speed_limit(
                angular_acceleration_limit, timestep
            )
        )

        # Saving constraints
        operational_constraints = OperationalConstraints()
        operational_constraints.weight = WEIGHT
        operational_constraints.max_rpm = max_rpm
        operational_constraints.max_thrust = max_thrust
        operational_constraints.max_z_torque = max_z_torque
        operational_constraints.hover_rpm = hover_rpm

        operational_constraints.speed_limit = speed_limit
        operational_constraints.acceleration_limit = acceleration_limit

        operational_constraints.angular_speed_limit = angular_speed_limit
        operational_constraints.angular_acceleration_limit = angular_acceleration_limit

        operational_constraints.gnd_eff_h_clip = gnd_eff_h_clip
        operational_constraints.max_xy_torque = max_xy_torque

        return operational_constraints

class DroneURDFHandler:
    """Handler for drone's URDF files and related operations."""

    def __init__(
        self, drone_model: DroneModel, environment_parameters: EnvironmentParameters
    ):
        self.environment_parameters = environment_parameters
        self.drone_model = drone_model

    def load_model(self, initial_position, initial_quaternion):
        """Load the drone model and return its ID and parameters."""

        drone_model = self.drone_model
        environment_parameters = self.environment_parameters
        client_id = environment_parameters.client_id
        urdf_file_path = DroneURDFHandler._create_path(drone_model=drone_model)
        tree = etxml.parse(urdf_file_path)
        root = tree.getroot()

        quadcopter_id = DroneURDFHandler._load_to_pybullet(
            initial_position, initial_quaternion, urdf_file_path, client_id
        )
        quadcopter_specs = DroneURDFHandler._load_parameters(
            root, environment_parameters
        )

        return quadcopter_id, quadcopter_specs

    @staticmethod
    def _create_path(drone_model: DroneModel) -> str:
        """Generate the path for the given drone model's URDF file."""

        # base_path = Path(os.getcwd()).parent
        base_path = Path(__file__).resolve().parent.parent.parent
        urdf_name = f"{drone_model.value}.urdf"
        return str(base_path / "assets" / urdf_name)

    @staticmethod
    def _load_to_pybullet(position, attitude, urdf_file_path, client_id):
        """Load the drone model into pybullet and return its ID."""
        quarternion = p.getQuaternionFromEuler(attitude)
        return p.loadURDF(
            urdf_file_path,
            position,
            quarternion,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client_id,
        )

    @staticmethod
    def _load_parameters(
        root, environment_parameters: EnvironmentParameters
    ) -> QuadcopterSpecs:
        """Loads parameters from an URDF file.
        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """

        URDF_TREE = root  # self.root
        M = float(URDF_TREE[1][0][1].attrib["value"])
        L = float(URDF_TREE[0].attrib["arm"])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])
        IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
        IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
        IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib["kf"])
        KM = float(URDF_TREE[0].attrib["km"])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
        PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
        DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
        DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])

        WEIGHT = M * environment_parameters.G
        return QuadcopterSpecs(
            M=M,
            L=L,
            THRUST2WEIGHT_RATIO=THRUST2WEIGHT_RATIO,
            J=J,
            J_INV=J_INV,
            KF=KF,
            KM=KM,
            COLLISION_H=COLLISION_H,
            COLLISION_R=COLLISION_R,
            COLLISION_Z_OFFSET=COLLISION_Z_OFFSET,
            MAX_SPEED_KMH=MAX_SPEED_KMH,
            GND_EFF_COEFF=GND_EFF_COEFF,
            PROP_RADIUS=PROP_RADIUS,
            DRAG_COEFF=DRAG_COEFF,
            DW_COEFF_1=DW_COEFF_1,
            DW_COEFF_2=DW_COEFF_2,
            DW_COEFF_3=DW_COEFF_3,
            WEIGHT=WEIGHT,
        )