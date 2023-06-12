import time

import curses
import random

import numpy as np
import pybullet as p
import pybullet_data


import gymnasium as gym
from gymnasium import spaces

from modules.factories.drone_factory import DroneFactory, Drone
from modules.factories.loiteringmunition_factory import LoiteringMunitionFactory, LoiteringMunition
from modules.factories.loyalwingman_factory import LoyalWingmanFactory, LoyalWingman

from modules.environments.environment_models import EnvironmentParameters


class DemoEnvironment(gym.Env):

    metadata = {"render.modes": ["human"]}

    ################################################################################

    def __init__(
        self,
        simulation_frequency: int = 240,
        rl_frequency: int = 60,
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

        #### Demo Debug Setup ##################
        self.setup_demo_lidar_log()

    def setup_factories(self):
        self.lwingman_factory: DroneFactory = self.setup_lw_factory()
        self.lmunition_factory: DroneFactory = self.setup_lm_factory()

    def setup_lw_factory(self) -> DroneFactory:
        return self.setup_drone_factory(LoyalWingmanFactory)

    def setup_lm_factory(self) -> DroneFactory:
        return self.setup_drone_factory(LoiteringMunitionFactory)

    def setup_drone_factory(self, factory_s) -> DroneFactory:
        factory: DroneFactory = factory_s()

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

    def setup_drones(self, factory: DroneFactory, quantity: int = 1) -> np.array:
        drones = np.array([], dtype=Drone)

        for _ in range(quantity):
            random_position = self.gen_random_position()
            factory.set_initial_position(random_position)
            drone = factory.create()
            drone.update_kinematics()

            drones = np.append(drones, drone)

        return drones

    def setup_loyalwingmen(self, quantity: int = 1) -> np.array:
        drones: np.array = self.setup_drones(self.lwingman_factory, quantity)
        drones.astype(LoyalWingman)
        return drones

    def setup_loiteringmunition(self, quantity: int = 1) -> np.array:
        drones: np.array = self.setup_drones(self.lmunition_factory, quantity)
        drones.astype(LoiteringMunition)
        return drones

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

        lw: LoyalWingman = self.loyalwingmen[0]
        lm: LoiteringMunition = self.loitering_munitions[0]

        return lw.observation(loyalwingmen=self.loyalwingmen, loitering_munitions=self.loitering_munitions)

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
        target_position = self.loitering_munitions[0].kinematics.position
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

        target_position = self.loitering_munitions[0].kinematics.position
        target_velocity = self.loitering_munitions[0].kinematics.velocity

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

    #####################################################################################################
    # Log
    #####################################################################################################

    def format_list(self, list_of_values):
        return str.join(" ", ["%0.2f".center(5) % i for i in list_of_values])

    def setup_demo_lidar_log(self):

        max_distance: float = 5
        resolution: float = 0.01

        self.loyalwingmen[0].set_lidar_parameters(
            max_distance=max_distance, resolution=resolution)

        return max_distance, resolution

    def generate_lidar_log(self):
        lw_kinematics = self.loyalwingmen[0].kinematics
        lm_kinematics = self.loitering_munitions[0].kinematics

        lw_position = lw_kinematics.position
        lm_position = lm_kinematics.position

        distance = np.linalg.norm(lm_position - lw_position)
        direction = (lm_position - lw_position) / distance

        max_distance, resolution = self.setup_demo_lidar_log()
        obs = np.round(self.observation, 2)

        text = "The Demo Environment reduces the lidar resolution to be able to log it"
        text += "\n"
        text += f'LiDAR Max distance: {max_distance}, Resolution: {resolution}'
        text += "\n"
        text += "LoyalWingman position:"
        text += "({:.2f}, {:.2f}, {:.2f})".format(
            lw_position[0], lw_position[1], lw_position[2])
        text += "\n"
        text += "LoiteringMunition position:"
        text += "({:.2f}, {:.2f}, {:.2f})".format(
            lm_position[0], lm_position[1], lm_position[2])
        text += "\n"
        text += "direction: " + self.format_list(direction) + "\n"
        text += "distance: {:.2f}".format(distance) + "\n"
        text += "reward: {:.2f}".format(self.last_reward) + "\n"
        text += "action: " + self.format_list(self.last_action) + "\n"
        text += "Lidar View"
        text += "\n"
        text += "\n"
        text += np.array2string(obs)

        return text

    def show_lidar_log(self):
        # print(self.observation)
        text = self.generate_lidar_log()
        # stdscr = self.init_curses()
        stdscr = curses.initscr()
        stdscr.clear()

        try:
            stdscr.addstr(0, 0, text)
            stdscr.refresh()
        except curses.error:
            pass

    def init_curses(self):
        if not self.curses_activated:
            stdscr = curses.initscr()
            self.curses_activated = True

        return stdscr
