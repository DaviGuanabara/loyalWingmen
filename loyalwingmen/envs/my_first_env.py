import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image

# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gym
from utils.enums import DroneModel, Physics, ImageType
from gym import spaces

from dataclasses import dataclass, fields, asdict


from typing import NamedTuple

from utils.managers.agent_manager import (
    apply_velocity_action,
    update_kinematics,
    gen_drone,
    collect_kinematics,
)

# from utils.agent_factory import gen_drone


# TODO: Fazer uma classe só com as constantes.
# TODO: Rever o agent_factory e o agent_manager.
# TODO: normalizar a observação
# TODO: o drone está se aproximando no target, mas algo está quebrado, pois a distancia se aproxima de zero enquanto que a reward diminui absurdamente.
# Ele tá ficando mais longe, mas a distancia está aumentando.
# TODO em todo canto está a posição do alvo. Deveria estar em só um local e ser chamado. Tenho que usar o obstacle manager para isso, algo parecido com o drone manager.
# Talvez unir os dois managers


# Imutável
class EnvParameters(NamedTuple):
    G: float
    NEIGHBOURHOOD_RADIUS: float


class MyFirstEnv(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {"render.modes": ["human"]}

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        GUI: bool = False,
    ):
        """Initialization of a generic aviary environment.
        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.
        """
        #### Constants #############################################
        self.envparameters = EnvParameters(G=9.8, NEIGHBOURHOOD_RADIUS=np.inf)

        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.PHYSICS = physics
        self.URDF = str(drone_model.value + ".urdf")

        self.step_counter = 0
        self.RESET_TIME = time.time()

        # self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        #### Connect to PyBullet ###################################

        #### With debug GUI ########################################
        # p.connect(p.GUI, options="--opengl2")

        if GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            ]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=-30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.CLIENT,
            )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            # print("viewMatrix", ret[2])
            # print("projectionMatrix", ret[3])

        else:
            self.CLIENT = p.connect(p.DIRECT)

        p.setGravity(0, 0, -self.envparameters.G, physicsClientId=self.CLIENT)

        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        # p.setTimeStep(1.0 / 240, physicsClientId=self.CLIENT)
        # p.setTimeStep(1.0, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT
        )

        #### Without debug GUI #####################################
        # self.CLIENT = p.connect(p.DIRECT)
        #### Uncomment the following line to use EGL Render Plugin #
        # Instead of TinyRender (CPU-based) in PYB's Direct mode
        # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)

        #### Set initial poses #####################################

        # self._parseURDFParameters(drone_model.value + ".urdf")
        # urdf_file_path = str("assets/" + drone_model.value + ".urdf")
        # self.load_agent(urdf_file_path)

        self.drone = gen_drone(
            client_id=self.CLIENT,
            urdf_file_path="assets/" + drone_model.value + ".urdf",
            gravity_acceleration=self.envparameters.G,
        )

        update_kinematics(self.CLIENT, self.drone)

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        # self._housekeeping()
        #### Update and store the drones kinematic information #####
        # self._updateAndStoreKinematicInformation()

        #### Set PyBullet's parameters #############################

        ################################################################################

    def reset(self):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        self.RESET_TIME = time.time()

        #### Housekeeping ##########################################

        self._housekeeping()

        #### Update and store the drones kinematic information #####
        # self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        # self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()

    ################################################################################

    def step(self, action):
        """Advances the environment by one simulation step.
        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.
        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.
        """

        apply_velocity_action(self.CLIENT, self.drone, action)
        update_kinematics(self.CLIENT, self.drone)
        p.stepSimulation()

        self.step_counter += 1

        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        return obs, reward, done, info

    ################################################################################

    def close(self):
        """Terminates the environment."""
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.
        Returns
        -------
        int:
            The PyBullet Client Id.
        """
        return self.CLIENT

    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.
        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.
        """
        return self.DRONE_IDS

    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Initialize the drones kinemaatic information ##########

        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.envparameters.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(1.0 / 240, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT
        )

        self.drone = gen_drone(
            client_id=self.CLIENT,
            urdf_file_path="assets/" + "cf2x.urdf",  # drone_model.value + ".urdf"
            initial_position=np.ones(3),
            gravity_acceleration=self.envparameters.G,
        )

        update_kinematics(self.CLIENT, self.drone)

        # print("load drones position:", self.INIT_XYZS[0,:])
        # self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
        #                                      self.INIT_XYZS[i, :],
        #                                      p.getQuaternionFromEuler(
        #                                          self.INIT_RPYS[i, :]),
        #                                      flags=p.URDF_USE_INERTIA_FROM_FILE,
        #                                      physicsClientId=self.CLIENT
        #                                      ) for i in range(self.NUM_DRONES)])
        # if self.OBSTACLES:
        #    self._addObstacles()

    ################################################################################

    ## É necessário guardar a posição e o resto dos dados.
    ## Todo to do: Armezenar os dados em outra função
    def _updateKinematicInformation(self, client_id, drone_id):
        """Updates the drones kinematic information.
        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """

        position, quartenions = p.getBasePositionAndOrientation(
            drone_id, physicsClientId=client_id
        )
        angular_position = p.getEulerFromQuaternion(quartenions)
        velocity, angular_velocity = p.getBaseVelocity(
            drone_id, physicsClientId=client_id
        )

        return position, angular_position, velocity, angular_velocity

    ################################################################################

    # todo to do: horrível. Ele usa uma série de variáveis globais. Dá pra deixar um doido.
    def _getDroneStateVector(self, nth_drone):
        """Returns the state vector of the n-th drone.
        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.
        """
        state = np.hstack(
            [
                self.pos[nth_drone, :],
                self.quat[nth_drone, :],
                self.rpy[nth_drone, :],
                self.vel[nth_drone, :],
                self.ang_v[nth_drone, :],
                self.last_clipped_action[nth_drone, :],
            ]
        )
        return state.reshape(
            20,
        )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.
        """

        return spaces.Box(
            low=np.array([-1, -1, -1, 0]),  # Alternative action space, see PR #32
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """

        return spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32,
        )

    ################################################################################

    # def process_kinematics_to_state(self, kinematics):
    #    kinematics_dictionary = asdict(kinematics)
    #    kinematics_in_list = list(kinematics_dictionary.values())
    #    state = np.hstack(kinematics_in_list).reshape(16)

    #    return state.reshape(16)

    # TODO: preparar a observação
    def _computeObs(self):
        """Returns the current observation of the environment.
        Must be implemented in a subclass.
        """
        # raise NotImplementedError

        drone_kinematics = collect_kinematics(self.CLIENT, self.drone)
        # drone_state = self.process_kinematics_to_state(drone_kinematics)
        drone_position = drone_kinematics.position
        drone_velocity = drone_kinematics.velocity

        target_position = np.array([0.5, 0.5, 0.5])
        target_velocity = np.array([0, 0, 0])

        # print(kinematics)
        # time.sleep(0.3)

        distance = np.linalg.norm(target_position - drone_position)
        direction = (target_position - drone_position) / distance

        observation = np.hstack(
            (
                self._normalizePosition(drone_position),
                self._normalizeVelocity(drone_velocity),
                self._normalizePosition(target_position),
                self._normalizeVelocity(target_velocity),
                direction,
                self._normalizeDistance(distance),
            )
        ).reshape(16)

        return observation  # np.zeros(16).astype("float32")

    def _computeReward(self):
        """Computes the current reward value(s).
        Must be implemented in a subclass.
        """

        drone_position = self.drone.kinematics.position
        target = np.array([0.5, 0.5, 0.5])

        distance = np.linalg.norm(target - drone_position)
        # raise NotImplementedError
        return -1 * distance  # * self.step_counter

    def _computeDone(self):
        """Computes the current done value(s).
        Must be implemented in a subclass.
        """
        # raise NotImplementedError

        drone_kinematics = collect_kinematics(self.CLIENT, self.drone)

        drone_position = drone_kinematics.position
        drone_velocity = drone_kinematics.velocity

        target_position = np.array([0.5, 0.5, 0.5])
        target_velocity = np.array([0, 0, 0])
        distance = np.linalg.norm(target_position - drone_position)

        MAX_DISTANCE = 100
        ERROR = 0.01

        current = time.time()

        if current - self.RESET_TIME > 20:
            return True

        if distance > MAX_DISTANCE or distance < ERROR:
            return True

        # if drone_position[2] < 0:
        #    return True

        return False

    def _computeInfo(self):
        """Computes the current info dict(s).
        Must be implemented in a subclass.
        """
        # raise NotImplementedError
        return {}

    ################################################################################

    #####################################################################
    # Física
    #####################################################################

    def _normalizedActionToRPM(self, action):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.
        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.
        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.
        """
        if np.any(np.abs(action) > 1):
            print(
                "\n[ERROR] it",
                self.step_counter,
                "in BaseAviary._normalizedActionToRPM(), out-of-bound action",
            )
        # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`
        return np.where(
            action <= 0,
            (action + 1) * self.HOVER_RPM,
            self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM) * action,
        )

    #####################################################################################################
    ## Normalization
    #####################################################################################################

    def _normalizeVelocity(self, velocity: np.array):
        MAX_Velocity = 5
        normalized_velocity = (
            np.clip(velocity, -MAX_Velocity, MAX_Velocity) / MAX_Velocity
        )
        return normalized_velocity

    def _normalizePosition(self, position: np.array):
        MAX_X_Y = 100
        MAX_Z = 100

        normalized_position_x_y = np.clip(position[0:2], -MAX_X_Y, MAX_X_Y) / MAX_X_Y
        normalized_position_z = np.clip([position[2]], 0, MAX_Z) / MAX_Z

        normalized_position = np.concatenate(
            (normalized_position_x_y, normalized_position_z)
        )

        return normalized_position

    def _normalizeDistance(self, distance):
        MAX_DISTANCE = 100
        normalized_distance = (
            np.clip(distance, -MAX_DISTANCE, MAX_DISTANCE) / MAX_DISTANCE
        )
        return normalized_distance

    def _clipAndNormalizeState(self, observation):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)

        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped
