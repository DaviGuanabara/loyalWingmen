import os
from sys import platform
import time
import curses
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from pathlib import Path
from PIL import Image
import random
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import math

from modules.utils.enums import DroneModel, Physics, ImageType

import gymnasium as gym  # import gym
from gymnasium import spaces

from dataclasses import dataclass, fields, asdict
from modules.factories.obstacle_factory import ObstacleFactory

from typing import NamedTuple

from modules.factories.drone_factory import DroneFactory
from modules.environments.drone_and_cube_env import DroneAndCube
from modules.environments.environment_models import EnvironmentParameters
from modules.factories.accessories_factory import LiDAR


# TODO: Fazer uma classe só com as constantes.
# TODO: Rever o agent_factory e o agent_manager.
# TODO: normalizar a observação
# TODO: o drone está se aproximando no target, mas algo está quebrado, pois a distancia se aproxima de zero enquanto que a reward diminui absurdamente.
# Ele tá ficando mais longe, mas a distancia está aumentando.
# TODO em todo canto está a posição do alvo. Deveria estar em só um local e ser chamado. Tenho que usar o obstacle manager para isso, algo parecido com o drone manager.
# Talvez unir os dois managers

# TODO: mudar "drones" para Loyalwingmen. Com o loyalwingmen sendo vários drones com o decorator. Decorator como herança ?
class DroneLidar(DroneAndCube):
    """Base class for "drone aviary" Gym environments."""

    metadata = {"render.modes": ["human"]}

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        simulation_frequency: int = 240,
        rl_frequency: int = 15,
        # aggregate_phy_steps: int = 15,
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
            Whether to allocate the attributes needed by vision-based aviary subclasses.self.TIMESTEP
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.
        """

        #### Factories #############################################
        self.rl_action_activated = False
        self.drone_factory = DroneFactory()
        self.obstacle_factory = ObstacleFactory()

        if GUI:
            client_id = self.setup_pybulley_GUI()

        else:
            client_id = self.setup_pybullet_DIRECT()

        #### Constants #############################################

        self.setup_Parameteres(simulation_frequency, rl_frequency, client_id)

        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.PHYSICS = physics
        self.URDF = str(drone_model.value + ".urdf")

        self.step_counter = 0
        self.RESET_TIME = time.time()

        self.lidar: LiDAR = LiDAR(max_distance=3, resolution=.02)

        #### Housekeeping ##########################################
        self._housekeeping()

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

    def setup_pybullet_DIRECT(self):
        return p.connect(p.DIRECT)

    def setup_pybulley_GUI(self):
        client_id = p.connect(p.GUI)
        for i in [
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(i, 0, physicsClientId=client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=-30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=client_id,
        )
        ret = p.getDebugVisualizerCamera(physicsClientId=client_id)

        return client_id

    def setup_Parameteres(self, simulation_frequency, rl_frequency, client_id):
        self.environment_parameters = EnvironmentParameters(
            G=9.8,
            NEIGHBOURHOOD_RADIUS=np.inf,
            simulation_frequency=simulation_frequency,
            rl_frequency=rl_frequency,
            timestep_period=1 / simulation_frequency,
            aggregate_physics_steps=int(simulation_frequency / rl_frequency),
            client_id=client_id,
            max_distance=100,
            error=0.5,
        )

    def get_parameteres(self):
        return self.environment_parameters

    def setup_targets(
        self,
        number_of_targets: int = 1,
        initial_positions: np.array = np.array([[0, 0, 0]]),
        urdf_file_paths: np.array = np.array(["cube_small.urdf"]),
    ):
        assert (
            number_of_targets == initial_positions.shape[0]
            and number_of_targets == urdf_file_paths.shape[0]
        ), "number of targets not correctly adjusted"

        targets = np.array([])

        for i in range(number_of_targets):
            target = self.obstacle_factory.generate_extended_obstacle(
                environment_parameters=self.environment_parameters,
                urdf_file_path=urdf_file_paths[i],
                initial_position=np.array(initial_positions[i]),
                initial_angular_position=np.zeros(3),
            )

            targets = np.append(targets, target)

        return targets

    def apply_target_behavior(self, obstacle):
        obstacle.apply_frozen_behavior()
        # obstacle.apply_constant_velocity_behavior()

    def setup_drones(
        self,
        number_of_drones: int = 1,
        initial_positions: np.array = np.array([[1, 1, 1]]),
    ):
        # print(number_of_drones, initial_positions.size)
        assert (
            number_of_drones == initial_positions.shape[0]
        ), "number of drones not correctly adjusted"

        drones = np.array([])

        base_path = str(Path(os.getcwd()).parent.absolute())
        path = base_path + "\\" + "assets\\" + "cf2x.urdf"

        for i in range(number_of_drones):
            quadcopter = self.drone_factory.gen_extended_drone(
                environment_parameters=self.environment_parameters,
                urdf_file_path=path,  # "assets/" + "cf2x.urdf",  # drone_model.value + ".urdf"
                initial_position=initial_positions[i],
            )

            drones = np.append(drones, quadcopter)

        return drones

    def reset(self, seed=1):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        p.resetSimulation(
            physicsClientId=self.environment_parameters.client_id)
        self.RESET_TIME = time.time()

        #### Housekeeping ##########################################

        self._housekeeping()

        #### Update and store the drones kinematic information #####
        # self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        # self._startVideoRecording()
        #### Return the initial observation ########################
        # apagar computerInfo para versões do SB3 abaixo da 2.0.0
        return self._computeObs(), self._computeInfo()

    ################################################################################

    # TODO preciso fixar a posição do target
    def step(self, rl_action):
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

        # TODO: o rl_action está desabilitado até que eu consiga alinhar os aspectos da behavior_tree com o do RL. Até lá

        # É importante para que uma decisão da rede neural tenha realmente impacto
        for _ in range(self.environment_parameters.aggregate_physics_steps):
            # ainda não está pronto múltiplos drones.
            for drone in self.drones:
                velocity_action = rl_action

                self.last_action = velocity_action
                drone.apply_velocity_action(velocity_action)
                drone.update_kinematics()

            for target in self.targets:
                self.apply_target_behavior(target)
                target.update_kinematics()

            p.stepSimulation()

        self.step_counter += 1

        self.observation = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeDone()
        info = self._computeInfo()

        # return obs, reward, done, info
        # apagar False (Truncated) para versões do SB3 abaixo da 2.0.0
        return self.observation, reward, terminated, False, info

    ################################################################################

    def close(self):
        """Terminates the environment."""
        # if self.RECORD and self.GUI:
        #    p.stopStateLogging(
        #        self.VIDEO_ID, physicsClientId=self.environment_parameters.client_id
        #    )
        p.disconnect(physicsClientId=self.environment_parameters.client_id)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.
        Returns
        -------
        int:
            The PyBullet Client Id.
        """
        return self.environment_parameters.client_id

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

    def gen_random_position(self):
        x = random.choice([-1, 1]) * random.random() * 2
        y = random.choice([-1, 1]) * random.random() * 2
        z = random.choice([-1, 1]) * random.random() * 2

        return [x, y, z]

    def _housekeeping(self):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Initialize the drones kinemaatic information ##########

        #### Set PyBullet's parameters #############################
        p.setGravity(
            0,
            0,
            -self.environment_parameters.G,
            physicsClientId=self.environment_parameters.client_id,
        )
        p.setRealTimeSimulation(
            0, physicsClientId=self.environment_parameters.client_id
        )  # No Realtime Sync
        p.setTimeStep(
            self.environment_parameters.timestep_period,
            physicsClientId=self.environment_parameters.client_id,
        )  # EachStep takes self.TIMESTEP_PERIOD to execute

        # p.setTimeStep(1.0 / 240, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.environment_parameters.client_id,
        )

        # TODO: Arrumar um nome melhor. Tipo, extended_drones. Pq esses drones aí são o 'Drone' em si embrulhado com o DroneDecorator.

        initial_drone_position = self.gen_random_position()
        self.drones = self.setup_drones(
            number_of_drones=1, initial_positions=np.array([initial_drone_position])
        )

        # TODO: Arrumar um nome melhor. Tipo, extended_obstacles. Pq esses targets nem lembrar de obstacle lembra, muito menos do decorator do obstacle (ObstacleDecorator)
        initial_target_position = [0, 0, 0]  # self.gen_random_position()
        self.targets = self.setup_targets(
            number_of_targets=1, initial_positions=np.array([initial_target_position])
        )

        for i in range(self.drones.size):
            self.drones[i].update_kinematics()

        for i in range(self.targets.size):
            self.targets[i].update_kinematics()

        self.lidar.reset()

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

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.
        """
        """
        Por algum motivo inusitado, o stable baselines 3 quebra com
        o Box do gymnasium
        assert isinstance(self.action_space, supported_action_spaces), (
        Mensagem de erro: AssertionError: The algorithm only supports (<class 'gym.spaces.box.Box'>, 
        <class 'gym.spaces.discrete.Discrete'>, 
        <class 'gym.spaces.multi_discrete.MultiDiscrete'>, <class 'gym.spaces.multi_binary.MultiBinary'>) 
        as action spaces but Box([-1. -1. -1.  0.], 1.0, (4,), float32) was provided
        Então, tô puxando do gym antigo até eles ajeitarem isso (oficialmente, o SB3 dá suporte).

        https://stackoverflow.com/questions/75832713/stable-baselines-3-support-for-farama-gymnasium
        """
        # a workaround to work with gymnasium
        # return old_gym_Box(
        #    low=np.array([-1, -1, -1, 0]),  # Alternative action space, see PR #32
        #    high=np.array([1, 1, 1, 1]),
        #    shape=(4,),
        #    dtype=np.float32,
        # )

        return spaces.Box(
            # Alternative action space, see PR #32
            low=np.array([-1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32,
        )

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """
        # a workaround to work with gymnasium

        # return old_gym_Box(
        #    low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0]),
        #    high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        #    dtype=np.float32,
        # )
        return spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1,
                         0, -1, -1, -1, -1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32,
        )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.
        Must be implemented in a subclass.
        """

        self.lidar.reset()
        drone_position = self.drones[0].gadget.kinematics.position

        for target in self.targets:
            target_position = target.gadget.kinematics.position

            self.lidar.add_position(
                loitering_munition_position=target_position, current_position=drone_position)

        return self.lidar.get_matrix()

    def _computeReward(self):
        """Computes the current reward value(s).
        Must be implemented in a subclass.
        """

        # TODO adicionar o Survivor Bonus
        # TODO adicionar penalidade por morrer.
        # TODO adicionar bonus por chegar no alvo.\

        # max_distance = self.environment_parameters.max_distance
        penalty = 0
        bonus = 0

        drone_position = self.drones[0].gadget.kinematics.position
        target_position = self.targets[0].gadget.kinematics.position
        distance = np.linalg.norm(target_position - drone_position)

        if distance > self.environment_parameters.max_distance:
            penalty += 100_000

        if distance < self.environment_parameters.error:
            bonus += 10_000 * \
                (self.environment_parameters.error - 1 * distance)

        self.last_reward = (5) - 1 * distance + bonus - penalty

        return self.last_reward

    def _computeDone(self):
        """Computes the current done value(s).
        Must be implemented in a subclass.
        """

        drone_position = self.drones[0].gadget.kinematics.position
        drone_velocity = self.drones[0].gadget.kinematics.velocity

        target_position = self.targets[0].gadget.kinematics.position
        target_velocity = self.targets[0].gadget.kinematics.velocity

        distance = np.linalg.norm(target_position - drone_position)

        current = time.time()

        if current - self.RESET_TIME > 20:
            return True

        if (
            np.linalg.norm(
                drone_position) > self.environment_parameters.max_distance
            or
            np.linalg.norm(
                target_position) > self.environment_parameters.max_distance
            or
            distance < self.environment_parameters.error
        ):
            return True

        return False

    def _computeInfo(self):
        """Computes the current info dict(s).
        Must be implemented in a subclass.
        """
        # raise NotImplementedError
        return {}

    #####################################################################################################
    # Normalization
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

        normalized_position_x_y = np.clip(
            position[0:2], -MAX_X_Y, MAX_X_Y) / MAX_X_Y
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

    def format_list(self, list_of_values):
        return str.join(" ", ["%0.2f".center(5) % i for i in list_of_values])

    def generate_log(self):
        drone_kinematics = self.drones[0].gadget.kinematics  # .position
        # drone_state = self.process_kinematics_to_state(drone_kinematics)
        drone_position = drone_kinematics.position
        drone_velocity = drone_kinematics.velocity

        target_kinematics = self.targets[0].gadget.kinematics
        target_position = target_kinematics.position
        target_velocity = target_kinematics.velocity

        # print(kinematics)
        # time.sleep(0.3)

        distance = np.linalg.norm(target_position - drone_position)
        direction = (target_position - drone_position) / distance

        text = ""
        text += (
            "WARNING: RL ACTION IS DISABLED"
            if not self.rl_action_activated
            else "WARNING: RL ACTION IS ENABLED"
        )

        text += "\n"
        text += "drone_position: " + self.format_list(drone_position) + "\n"
        text += "drone_velocity: " + self.format_list(drone_velocity) + "\n"
        text += "target_position: " + self.format_list(target_position) + "\n"
        text += "target_velocity: " + self.format_list(target_velocity) + "\n"
        text += "direction: " + self.format_list(direction) + "\n"
        text += "distance: " + str(distance) + "\n"
        text += "reward: " + str(self.last_reward) + "\n"
        text += "action: " + self.format_list(self.last_action) + "\n"

        return text

    def show_log(self):
        text = self.generate_log()

        stdscr = curses.initscr()
        stdscr.addstr(0, 0, text)
        stdscr.refresh()

    def generate_lidar_log(self):

        obs = np.round(self.observation, 2)

        text = ""
        text += "\n"
        text += np.array2string(obs)

        return text

    def show_lidar_log(self):
        text = self.generate_lidar_log()
        # print(text)

        stdscr = curses.initscr()
        stdscr.clear()
        stdscr.addstr(0, 0, text)
        stdscr.refresh()
