import os
import platform
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


from modules.factories.drone_factory import DroneFactory, Drone
from modules.factories.loiteringmunition_factory import LoiteringMunitionFactory
from modules.factories.loyalwingman_factory import LoyalWingmanFactory

from modules.environments.environment_models import EnvironmentParameters


class DroneLidar2(gym.Env):

    metadata = {"render.modes": ["human"]}

    ################################################################################

    def __init__(
        self,
        simulation_frequency: int = 240,
        rl_frequency: int = 15,
        GUI: bool = False,
    ):

        #### client #############################################
        if GUI:
            client_id = self.setup_pybulley_GUI()

        else:
            client_id = self.setup_pybullet_DIRECT()

        #### Constants #############################################
        self.setup_Parameteres(simulation_frequency, rl_frequency, client_id)

        #### Options ###############################################
        self.RESET_TIME = time.time()

        #### Factories #############################################
        self.setup_factories()

        #### Housekeeping ##########################################
        self._housekeeping()

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

    def setup_factories(self):
        self.lwingman_factory: DroneFactory = self.setup_lw_factory()
        self.lmunition_factory: DroneFactory = self.setup_lw_factory()

    def setup_lw_factory(self) -> DroneFactory:
        factory: DroneFactory = LoyalWingmanFactory()

        factory.set_environment_parameters(self.environment_parameters)
        factory.set_initial_position(np.array([1, 1, 1]))
        factory.set_initial_angular_position(np.array([0, 0, 0]))

        return factory

    def setup_lw_factory(self) -> DroneFactory:
        factory: DroneFactory = LoyalWingmanFactory()

        factory.set_environment_parameters(self.environment_parameters)
        factory.set_initial_position(np.array([1, 1, 1]))
        factory.set_initial_angular_position(np.array([0, 0, 0]))

        return factory

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

    def apply_target_behavior(self, obstacle):
        obstacle.apply_frozen_behavior()
        # obstacle.apply_constant_velocity_behavior()

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
            for loyalwingman in self.loyalwingmen:
                velocity_action = rl_action

                self.last_action = velocity_action
                loyalwingman.apply_velocity_action(velocity_action)
                loyalwingman.update_kinematics()

            for loitering_munition in self.loitering_munitions:
                self.apply_target_behavior(loitering_munition)
                loitering_munition.update_kinematics()

            p.stepSimulation()

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
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.environment_parameters.client_id,
        )

        self.loyalwingmen = self.setup_loyalwingmen(1)
        self.loitering_munitions = self.setup_loiteringmunition(1)

    def setup_drones(self, factory: DroneFactory, quantity: int = 1):
        drones = np.array([], dtype=Drone)

        for _ in range(quantity):
            random_position = self.gen_random_position()
            factory.set_initial_position(random_position)
            drone = factory.create()
            drone.update_kinematics()

            drones = np.append(drones, drone)

        return drones

    def setup_loyalwingmen(self, quantity: int = 1) -> np.array:
        return self.setup_drones(self.lwingman_factory, quantity)

    def setup_loiteringmunition(self, quantity: int = 1) -> np.array:
        return self.setup_drones(self.lmunition_factory, quantity)

    ################################################################################

    def _actionSpace(self):

        return spaces.Box(
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

        drone_position = self.loyalwingmen[0].kinematics.position

        for loitering_munition in self.loitering_munitions:
            loitering_munition_position = loitering_munition.kinematics.position

            # self.lidar.add_position(
            #    loitering_munition_position=target_position,
            #    current_position=drone_position,
            # )

        return  # self.lidar.get_matrix()

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

        drone_position = self.loyalwingmen[0].kinematics.position
        target_position = self.loitering_munitions[0].gadget.kinematics.position
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

        drone_position = self.loyalwingmen[0].kinematics.position
        drone_velocity = self.loyalwingmen[0].kinematics.velocity

        target_position = self.loitering_munitions[0].gadget.kinematics.position
        target_velocity = self.loitering_munitions[0].gadget.kinematics.velocity

        distance = np.linalg.norm(target_position - drone_position)

        current = time.time()

        if current - self.RESET_TIME > 20:
            return True

        if (
            np.linalg.norm(
                drone_position) > self.environment_parameters.max_distance
            or np.linalg.norm(target_position)
            > self.environment_parameters.max_distance
            or distance < self.environment_parameters.error
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
