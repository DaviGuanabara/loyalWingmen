import numpy as np
import pybullet as p
import platform
import os
from pathlib import Path

import xml.etree.ElementTree as etxml
from typing import Tuple

from typing import Union

from .base.quadcopter import (
    Quadcopter,
    QuadcopterSpecs,
    OperationalConstraints,
    QuadcopterType,
    DroneModel,
)
from ..environments.dataclasses.environment_parameters import EnvironmentParameters
from .loyalwingman import LoyalWingman
from .loiteringmunition import LoiteringMunition
from enum import Enum, auto


class DroneType(Enum):
    LOYALWINGMAN = auto()
    LOITERINGMUNITION = auto()


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

        base_path = Path(os.getcwd()).parent
        urdf_name = f"{drone_model.value}.urdf"
        return str(base_path / "assets" / urdf_name)

    @staticmethod
    def _load_to_pybullet(position, attitude, urdf_file_path, client_id):
        """Load the drone model into pybullet and return its ID."""
        quarternion = p.quaternionFromEuler(attitude)
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


class QuadcopterFactory:
    def __init__(
        self,
        environment_parameters: EnvironmentParameters,
        drone_model: DroneModel = DroneModel.CF2X,
    ):
        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self.drone_model = drone_model
        self.drone_urdf_handler = DroneURDFHandler(
            self.drone_model, self.environment_parameters
        )
        self.environment_parameters = environment_parameters

        self.constructor = {
            DroneType.LOYALWINGMAN: LoyalWingman,
            DroneType.LOITERINGMUNITION: LoiteringMunition,
        }

        self.n_loyalwingmen = 0
        self.n_loiteringmunitions = 0

    def __compute_OperationalConstraints(self, parameters: QuadcopterSpecs):
        gravity_acceleration = self.environment_parameters.G
        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1

        L = parameters.L
        M = parameters.M
        KF = parameters.KF
        KM = parameters.KM
        PROP_RADIUS = parameters.PROP_RADIUS
        GND_EFF_COEFF = parameters.GND_EFF_COEFF
        THRUST2WEIGHT_RATIO = parameters.THRUST2WEIGHT_RATIO

        gravity = gravity_acceleration * M
        max_rpm = np.sqrt((THRUST2WEIGHT_RATIO * gravity) / (4 * KF))
        max_thrust = 4 * KF * max_rpm**2
        max_z_torque = 2 * KM * max_rpm**2
        hover_rpm = np.sqrt(gravity / (4 * KF))
        speed_limit = VELOCITY_LIMITER * parameters.MAX_SPEED_KMH * KMH_TO_MS
        gnd_eff_h_clip = (
            0.25
            * PROP_RADIUS
            * np.sqrt((15 * max_rpm**2 * KF * GND_EFF_COEFF) / max_thrust)
        )
        max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(2)

        operational_constraints = OperationalConstraints()
        operational_constraints.gravity = gravity
        operational_constraints.max_rpm = max_rpm
        operational_constraints.max_thrust = max_thrust
        operational_constraints.max_z_torque = max_z_torque
        operational_constraints.hover_rpm = hover_rpm
        operational_constraints.speed_limit = speed_limit
        operational_constraints.gnd_eff_h_clip = gnd_eff_h_clip
        operational_constraints.max_xy_torque = max_xy_torque

        return operational_constraints

    def load_quad_attributes(
        self, initial_position: np.ndarray, initial_angular_position: np.ndarray
    ) -> Tuple[
        int,
        DroneModel,
        QuadcopterSpecs,
        OperationalConstraints,
        EnvironmentParameters,
    ]:
        id, parameters = self.drone_urdf_handler.load_model(
            initial_position, initial_angular_position
        )
        operational_constraints = self.__compute_OperationalConstraints(parameters)
        environment_parameters = self.environment_parameters

        return (
            id,
            self.drone_model,
            parameters,
            operational_constraints,
            environment_parameters,
        )

    def create(
        self,
        type: DroneType,
        position: np.ndarray,
        ang_position: np.ndarray,
        quadcopter_name: str = "",
    ) -> Union[Quadcopter, LoyalWingman, LoiteringMunition, None]:
        quad_constructor = self.constructor.get(type)

        i = (
            self.n_loyalwingmen
            if type == DroneType.LOYALWINGMAN
            else self.n_loiteringmunitions
        )
        quadcopter_name = quadcopter_name or f"{self.drone_model.name}_{type.name}_{i}"

        if not quad_constructor:
            raise ValueError(f"Invalid drone type: {type}")

        attributes = self.load_quad_attributes(position, ang_position)
        return quad_constructor(*attributes, quadcopter_name=quadcopter_name)

    def create_loyalwingman(
        self, position: np.ndarray, ang_position: np.ndarray, quadcopter_name: str = ""
    ) -> LoyalWingman:
        i = self.n_loyalwingmen + 1
        self.n_loyalwingmen = i

        quadcopter_name = (
            quadcopter_name
            or f"{self.drone_model.name}_{QuadcopterType.LOYALWINGMAN}_{i}"
        )

        attributes = self.load_quad_attributes(position, ang_position)
        return LoyalWingman(*attributes, quadcopter_name=quadcopter_name)

    def create_loiteringmunition(
        self, position: np.ndarray, ang_position: np.ndarray, quadcopter_name: str = ""
    ) -> LoiteringMunition:
        i = self.n_loiteringmunitions + 1
        self.n_loiteringmunitions = i

        quadcopter_name = (
            quadcopter_name
            or f"{self.drone_model.name}_{QuadcopterType.LOITERINGMUNITION}_{i}"
        )

        attributes = self.load_quad_attributes(position, ang_position)
        return LoiteringMunition(*attributes, quadcopter_name=quadcopter_name)
