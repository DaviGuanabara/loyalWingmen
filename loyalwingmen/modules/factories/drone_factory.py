import numpy as np
import pybullet as p
import pybullet_data

from modules.control.DSLPIDControl import DSLPIDControl
from modules.utils.enums import DroneModel, Physics, ImageType

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass, field
from typing import NamedTuple
from modules.factories.factory_models import (
    Kinematics,
    Parameters,
    Drone,
    Drone_Informations,
)


from modules.decorators.drone_decorator import DroneDecorator


class DroneFactory:
    def __init__(self):
        pass

    def compute_informations(self, parameters: Parameters, gravity_acceleration: float):
        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1  # 0.03

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
        max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(
            2
        )  # Ajustado para Model CF2X

        informations = Drone_Informations()
        informations.gravity = gravity
        informations.max_rpm = max_rpm
        informations.max_thrust = max_thrust
        informations.max_z_torque = max_z_torque
        informations.hover_rpm = hover_rpm
        informations.speed_limit = speed_limit
        informations.gnd_eff_h_clip = gnd_eff_h_clip
        informations.max_xy_torque = max_xy_torque

        return informations

    def _parseURDFParameters(self, urdf_file_path):
        """Loads parameters from an URDF file.
        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        URDF_TREE = etxml.parse(urdf_file_path).getroot()
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
        return Parameters(
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
        )

    def load_agent(
        self,
        client_id: int,
        urdf_file_path: str,
        initial_position=np.ones(3),
        initial_angular_position=np.zeros(3),
    ):
        return p.loadURDF(
            urdf_file_path,
            initial_position,
            p.getQuaternionFromEuler(initial_angular_position),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client_id,
        )

    def gen_extended_drone(
        self,
        environment_parameters,
        urdf_file_path: str,
        initial_position: np.array = np.ones(3),
        initial_angular_position: np.array = np.zeros(3),
    ):
        parameters = self._parseURDFParameters(urdf_file_path)

        kinematics = Kinematics(
            position=initial_position, angular_position=initial_angular_position
        )
        id = self.load_agent(
            environment_parameters.client_id,
            urdf_file_path,
            initial_position,
            initial_angular_position,
        )
        informations = self.compute_informations(
            parameters, gravity_acceleration=environment_parameters.G
        )

        drone = Drone()
        drone.id = id
        drone.parameters = parameters
        drone.kinematics = kinematics
        drone.informations = informations

        extended_drone = DroneDecorator(
            drone=drone, environment_parameters=environment_parameters
        )

        return extended_drone
