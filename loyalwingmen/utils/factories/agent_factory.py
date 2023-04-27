import numpy as np
import pybullet as p
import pybullet_data

from control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple


# Mutável
@dataclass
class Kinematics:
    position: np.array = np.zeros(3)
    angular_position: np.array = np.zeros(3)
    quaternions: np.array = np.zeros(4)
    velocity: np.array = np.zeros(3)
    angular_velocity: np.array = np.zeros(3)


# self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
# Imutável
class Parameters(NamedTuple):
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: float
    J_INV: float
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: float
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float


class Drone_informations:
    speed_limit: float = 0
    gravity: float = 0
    max_rpm: float = 0
    max_thrust: float = 0

    max_z_torque: float = 0
    hover_rpm: float = 0
    speed_limit: float = 0
    gnd_eff_h_clip: float = 0

    max_xy_torque: float = 0


# Imutável
class Drone:  # (NamedTuple):
    id: int
    parameters: Parameters
    kinematics: Kinematics
    informations: Drone_informations
    control: DSLPIDControl = DSLPIDControl(drone_model=DroneModel.CF2X)


def compute_informations(parameters: Parameters, gravity_acceleration: float):
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
    max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(2)  # Ajustado para Model CF2X

    informations = Drone_informations()
    informations.gravity = gravity
    informations.max_rpm = max_rpm
    informations.max_thrust = max_thrust
    informations.max_z_torque = max_z_torque
    informations.hover_rpm = hover_rpm
    informations.speed_limit = speed_limit
    informations.gnd_eff_h_clip = gnd_eff_h_clip
    informations.max_xy_torque = max_xy_torque

    return informations


def _parseURDFParameters(urdf_file_path):
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


def gen_drone(
    client_id: int,
    urdf_file_path: str,
    initial_position: np.array = np.ones(3),
    initial_angular_position: np.array = np.zeros(3),
    gravity_acceleration: float = 9.8,
):
    parameters = _parseURDFParameters(
        urdf_file_path
    )  # TODO mudar o nome, pois ele é específico do drone, e não é genérico, como o nome dá a entender.
    kinematics = Kinematics(
        position=initial_position, angular_position=initial_angular_position
    )
    id = load_agent(
        client_id, urdf_file_path, initial_position, initial_angular_position
    )
    informations = compute_informations(
        parameters, gravity_acceleration=gravity_acceleration
    )

    drone = Drone()
    drone.id = id
    drone.parameters = parameters
    drone.kinematics = kinematics
    drone.informations = informations

    return drone
