import numpy as np
import pybullet as p
import pybullet_data

import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image


from dataclasses import dataclass
from typing import NamedTuple

from utils.factories.agent_factory import Kinematics
from utils.factories.agent_factory import Drone

from control.DSLPIDControl import DSLPIDControl
from utils.factories.agent_factory import gen_drone

################################################################################
## Kinematics
################################################################################


def store_kinematics(client_id, drone, kinematics):
    drone.kinematics = kinematics


def update_kinematics(client_id, drone):
    kinematics = collect_kinematics(client_id, drone)
    store_kinematics(client_id, drone, kinematics)


def collect_kinematics(client_id, drone):
    position, quaternions = p.getBasePositionAndOrientation(
        drone.id, physicsClientId=client_id
    )
    angular_position = p.getEulerFromQuaternion(quaternions)
    velocity, angular_velocity = p.getBaseVelocity(drone.id, physicsClientId=client_id)

    kinematics = Kinematics(
        position=np.array(position),
        angular_position=np.array(angular_position),
        quaternions=np.array(quaternions),
        velocity=np.array(velocity),
        angular_velocity=np.array(angular_velocity),
    )

    return kinematics  # position, angular_position, velocity, angular_velocity


################################################################################
## Action
################################################################################


def _preprocessAction(action, drone: Drone, timestep: int):
    """Pre-processes the action passed to `.step()` into motors' RPMs.
    Parameter `action` is processed differenly for each of the different
    action types: `action` can be of length 1, 3, 4, or 6 and represent
    RPMs, desired thrust and torques, the next target position to reach
    using PID control, a desired velocity vector, new PID coefficients, etc.
    Parameters
    ----------
    action : ndarray
        The input action for each drone, to be translated into RPMs.
    Returns
    -------
    ndarray
        (4,)-shaped array of ints containing to clipped RPMs
        commanded to the 4 motors of each drone.
    """

    speed_limit = drone.informations.speed_limit

    position = drone.kinematics.position
    quaternions = drone.kinematics.quaternions
    angular_position = drone.kinematics.angular_position

    velocity = drone.kinematics.velocity
    angular_velocity = drone.kinematics.angular_velocity

    if np.linalg.norm(action[:3]) != 0:
        v_unit_vector = action[:3] / np.linalg.norm(action[:3])

    else:
        v_unit_vector = np.zeros(3)

    rpm, _, _ = drone.control.computeControl(
        control_timestep=timestep,
        cur_pos=np.array(position),
        cur_quat=quaternions,
        cur_vel=velocity,
        cur_ang_vel=angular_velocity,
        # same as the current position
        target_pos=np.array(position),
        target_rpy=[0, 0, angular_position[2]],  # angular_position,  # keep current yaw
        # target the desired velocity vector
        target_vel=speed_limit * np.abs(action[3]) * v_unit_vector,
    )

    return rpm


def apply_velocity_action(client_id, drone: Drone, action, freq: int = 240):
    timestep = 1.0 / freq
    rpm = _preprocessAction(action, drone, timestep)
    _physics(client_id, drone, rpm)


def _physics(client_id, drone: Drone, rpm):
    """Base PyBullet physics implementation.
    Parameters
    ----------
    rpm : ndarray
        (4)-shaped array of ints containing the RPMs values of the 4 motors.
    nth_drone : int
        The ordinal number/position of the desired drone in list self.DRONE_IDS.
    """

    drone_id = drone.id
    KF = drone.parameters.KF
    KM = drone.parameters.KM

    forces = np.array(rpm**2) * KF
    torques = np.array(rpm**2) * KM
    z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

    for i in range(4):
        p.applyExternalForce(
            drone_id,
            i,
            forceObj=[0, 0, forces[i]],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=client_id,
        )
    p.applyExternalTorque(
        drone_id,
        4,
        torqueObj=[0, 0, z_torque],
        flags=p.LINK_FRAME,
        physicsClientId=client_id,
    )
