import numpy as np
import pybullet as p
import platform
import os
from pathlib import Path

import xml.etree.ElementTree as etxml
from typing import Union, Tuple

from .components.base.quadcopter import (
    Quadcopter,
    FlightStateDataType,
    QuadcopterSpecs,
    OperationalConstraints,
    QuadcopterType,
    DroneModel,
    CommandType,
)

from ..environments.helpers.environment_parameters import EnvironmentParameters
from .loyalwingman import LoyalWingman
from .loiteringmunition import LoiteringMunition, LoiteringMunitionBehavior
from enum import Enum, auto


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


class QuadcopterFactory:
    def __init__(
        self,
        environment_parameters: EnvironmentParameters,
        drone_model: DroneModel = DroneModel.CF2X,
    ):
        self.environment_parameters = environment_parameters
        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self.drone_model = drone_model
        self.drone_urdf_handler = DroneURDFHandler(
            self.drone_model, self.environment_parameters
        )

        self.n_loyalwingmen = 0
        self.n_loiteringmunitions = 0

    def load_quad_attributes(
        self,
        initial_position: np.ndarray,
        initial_angular_position: np.ndarray,
        quadcopter_type: QuadcopterType,
        quadcopter_name: str,
    ) -> Tuple[
        int,
        DroneModel,
        QuadcopterSpecs,
        OperationalConstraints,
        EnvironmentParameters,
        str,
    ]:
        id, parameters = self.drone_urdf_handler.load_model(
            initial_position, initial_angular_position
        )

        operational_constraints = OperationalConstraintsCalculator.compute(
            parameters, self.environment_parameters
        )

        environment_parameters = self.environment_parameters
        quadcopter_full_name = self._gen_quadcopter_name(
            quadcopter_type, quadcopter_name
        )

        return (
            id,
            self.drone_model,
            parameters,
            operational_constraints,
            environment_parameters,
            quadcopter_full_name,
        )

    def create_loyalwingman(
        self,
        position: np.ndarray,
        ang_position: np.ndarray,
        command_type: CommandType = CommandType.VELOCITY_DIRECT,
        quadcopter_name: str = "Drone",
    ) -> LoyalWingman:
        attributes = self.load_quad_attributes(
            position, ang_position, QuadcopterType.LOYALWINGMAN, quadcopter_name
        )
        return LoyalWingman(*attributes, command_type=command_type)

    def create_loiteringmunition(
        self,
        position: np.ndarray,
        ang_position: np.ndarray,
        command_type: CommandType = CommandType.VELOCITY_DIRECT,
        quadcopter_name: str = "Drone",
    ) -> LoiteringMunition:
        attributes = self.load_quad_attributes(
            position, ang_position, QuadcopterType.LOITERINGMUNITION, quadcopter_name
        )
        return LoiteringMunition(*attributes, command_type=command_type)

    def _gen_quadcopter_name(
        self, quadcopter_type: QuadcopterType, quadcopter_name: str = "Drone"
    ):
        if quadcopter_type == QuadcopterType.LOYALWINGMAN:
            return f"{quadcopter_type}_{self.n_loyalwingmen}_{quadcopter_name}"

        if quadcopter_type == QuadcopterType.LOITERINGMUNITION:
            return f"{quadcopter_type}_{self.n_loiteringmunitions}_{quadcopter_name}"

        return f"{quadcopter_type}_{quadcopter_name}"