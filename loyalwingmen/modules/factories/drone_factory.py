import numpy as np
import pybullet as p
import platform
import os
from pathlib import Path

import xml.etree.ElementTree as etxml
from typing import Tuple

from modules.control.DSLPIDControl import DSLPIDControl
from modules.utils.enums import DroneModel
from modules.interfaces.factory_interface import IDroneFactory
from modules.models.drone import Drone, Parameters, Kinematics, Informations, EnvironmentParameters


class DroneFactory(IDroneFactory):
    def __init__(self, environment_parameters: EnvironmentParameters, drone_model: DroneModel = DroneModel.CF2X,
                 initial_position: np.array = np.ones((3,)), initial_angular_position: np.array = np.zeros((3,))):

        self.set_drone_model(drone_model)
        self.set_environment_parameters(environment_parameters)
        self.set_initial_position(initial_position)
        self.set_initial_angular_position(initial_angular_position)

    # =================================================================================================================
    # Private
    # =================================================================================================================

    def __compute_kinematics(self) -> Kinematics:
        return Kinematics(
            position=self.initial_position, angular_position=self.initial_angular_position
        )

    def __compute_parameters(self):
        urdf_file_path = self.urdf_file_path
        return self.__parseURDFParameters(urdf_file_path)

    def __compute_informations(self, parameters: Parameters):

        gravity_acceleration = self.environment_parameters.G
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

        informations = Informations()
        informations.gravity = gravity
        informations.max_rpm = max_rpm
        informations.max_thrust = max_thrust
        informations.max_z_torque = max_z_torque
        informations.hover_rpm = hover_rpm
        informations.speed_limit = speed_limit
        informations.gnd_eff_h_clip = gnd_eff_h_clip
        informations.max_xy_torque = max_xy_torque

        return informations

    def __parseURDFParameters(self, urdf_file_path):
        """Loads parameters from an URDF file.
        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        # urdf_file_path = self.urdf_file_path
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

    def __compute_control(self, model: DroneModel, parameters: Parameters, environment_parameters: EnvironmentParameters, urdf_file_path: str):

        return DSLPIDControl(model, parameters, environment_parameters, urdf_path=urdf_file_path)

    def __load_urdf(self):
        id = p.loadURDF(
            self.urdf_file_path,
            self.initial_position,
            self.initial_quaternion,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.environment_parameters.client_id,
        )

        return id

    def __compute_drone_model(self):
        return self.drone_model  # DroneModel.CF2X

    def __setup_urdf_file_path(self, drone_model: DroneModel = DroneModel.CF2X):
        urdf_name = drone_model.value + ".urdf"
        base_path = str(Path(os.getcwd()).parent.absolute())

        if platform.system() == "Windows":
            path = base_path + "\\" + "assets\\" + urdf_name  # "cf2x.urdf"

        else:
            path = base_path + "/" + "assets/" + urdf_name  # "cf2x.urdf"

        self.set_urdf_file_path(path)

    # =================================================================================================================
    # Public
    # =================================================================================================================

    def load_drone_attributes(self) -> Tuple[int, DroneModel, Parameters, Informations, Kinematics, DSLPIDControl, EnvironmentParameters]:
        id = self.__load_urdf()
        model = self.__compute_drone_model()
        parameters = self.__compute_parameters()
        informations = self.__compute_informations(parameters)
        kinematics = self.__compute_kinematics()
        control = self.__compute_control(
            model, parameters, self.environment_parameters, self.urdf_file_path)
        environment_parameters = self.environment_parameters

        return id, model, parameters, informations, kinematics, control, environment_parameters

    def set_drone_model(self, drone_model: DroneModel):
        self.drone_model = drone_model
        self.__setup_urdf_file_path(drone_model)

    def set_environment_parameters(self, environment_parameters: EnvironmentParameters):
        self.environment_parameters = environment_parameters

    def set_urdf_file_path(self, urdf_file_path: str):
        self.urdf_file_path = urdf_file_path

    def set_initial_position(self, initial_position: np.array):
        self.initial_position = initial_position

    def set_initial_angular_position(self, initial_angular_position: np.array):
        self.initial_angular_position = initial_angular_position
        self.initial_quaternion = p.getQuaternionFromEuler(
            initial_angular_position)

    def create(self) -> Drone:

        id, model, parameters, informations, kinematics, control, environment_parameters = self.load_drone_attributes()
        drone = Drone(id, model, parameters, informations, kinematics,
                      control, environment_parameters)

        return drone
