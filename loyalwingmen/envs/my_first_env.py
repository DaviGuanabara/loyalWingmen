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

from utils.enums import DroneModel, Physics, ImageType

import gymnasium as gym  # import gym
from gymnasium import spaces
from gym.spaces import Box as old_gym_Box

from dataclasses import dataclass, fields, asdict
from utils.managers.obstacle_manager import (
    generate_obstacle,
    apply_force,
    apply_frozen_behavior,
)

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
class EnvironmentParameters(NamedTuple):
    G: float
    NEIGHBOURHOOD_RADIUS: float
    simulation_frequency: int
    rl_frequency: int
    timestep_period: float
    aggregate_physics_steps: int
    client_id: int
    max_distance: float
    error: float


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

        # self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        #### Connect to PyBullet ###################################

        #### With debug GUI ########################################
        # p.connect(p.GUI, options="--opengl2")

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

        #### Housekeeping ##########################################
        self._housekeeping()

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        # print(type(self.action_space))
        # print(type(self.observation_space))

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
            error=0.2,
        )

        #### Set PyBullet's parameters #############################

        self.SIMULATION_FREQUENCY = simulation_frequency
        self.RL_FREQUENCY = rl_frequency

        self.TIMESTEP_PERIOD = 1 / self.SIMULATION_FREQUENCY

        self.AGGREGATE_PHY_STEPS = int(
            self.SIMULATION_FREQUENCY / self.RL_FREQUENCY
        )  # Ou seja, quantas vezes o .step do pybullet deve ser chamado para cada ação da rede neural

    def setup_targets(
        self,
        number_of_targets: int = 1,
        initial_positions: np.array = np.array([[1, 1, 1]]),
        urdf_file_paths: np.array = np.array(["cube_small.urdf"]),
    ):
        assert (
            number_of_targets == initial_positions.shape[0]
            and number_of_targets == urdf_file_paths.shape[0]
        ), "number of targets not correctly adjusted"

        targets = np.array([])

        for i in range(number_of_targets):
            target = generate_obstacle(
                client_id=self.environment_parameters.client_id,
                urdf_file_path=urdf_file_paths[i],
                initial_position=np.array(initial_positions[i]),
                initial_angular_position=np.zeros(3),
            )

            targets = np.append(targets, target)

        return targets

    def apply_target_behavior(self, obstacle):
        apply_frozen_behavior(self.environment_parameters, obstacle)

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

        for i in range(number_of_drones):
            quadcopter = gen_drone(
                client_id=self.environment_parameters.client_id,
                urdf_file_path="assets/" + "cf2x.urdf",  # drone_model.value + ".urdf"
                initial_position=initial_positions[i],
                gravity_acceleration=self.environment_parameters.G,
            )

            drones = np.append(drones, quadcopter)

        return drones

    def reset(self):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        p.resetSimulation(physicsClientId=self.environment_parameters.client_id)
        self.RESET_TIME = time.time()

        #### Housekeeping ##########################################

        self._housekeeping()

        #### Update and store the drones kinematic information #####
        # self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        # self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()  # , self._computeInfo()

    ################################################################################

    # TODO preciso fixar a posição do target
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

        # TODO usar o agregate physic steps para diminuir a taxa de atualização da rede neural.
        # Por exemplo, deixar o simulador a 240 hz, mas a rede neural a 15hz, para que
        # a ação seja "sentida". É como jogar em um monitor a 30hz. Um humano consegue tomar a decisão mesmo não estando
        # tomando uma decisão a cada milisegundo.

        for _ in range(
            self.AGGREGATE_PHY_STEPS
        ):  # É importante para que uma decisão da rede neural tenha realmente impacto
            apply_velocity_action(
                self.environment_parameters.client_id, self.drones[0], action
            )

            # TODO isso está quebrando não sei o motivo.
            self.apply_target_behavior(self.targets[0])

            p.stepSimulation()
            update_kinematics(self.environment_parameters.client_id, self.drones[0])

        self.step_counter += 1

        observation = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeDone()
        info = self._computeInfo()

        # return obs, reward, done, info
        return observation, reward, terminated, info

    ################################################################################

    def close(self):
        """Terminates the environment."""
        if self.RECORD and self.GUI:
            p.stopStateLogging(
                self.VIDEO_ID, physicsClientId=self.environment_parameters.client_id
            )
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

        self.drones = self.setup_drones(
            number_of_drones=1, initial_positions=np.array([[1, 1, 1]])
        )

        self.targets = self.setup_targets(
            number_of_targets=1, initial_positions=np.array([[0.5, 0.5, 0.5]])
        )

        for i in range(self.drones.size):
            update_kinematics(self.environment_parameters.client_id, self.drones[i])

        for i in range(self.targets.size):
            update_kinematics(self.environment_parameters.client_id, self.targets[i])

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
        return old_gym_Box(
            low=np.array([-1, -1, -1, 0]),  # Alternative action space, see PR #32
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32,
        )

        # return spaces.Box(
        #    low=np.array([-1, -1, -1, 0]),  # Alternative action space, see PR #32
        #    high=np.array([1, 1, 1, 1]),
        #    shape=(4,),
        #    dtype=np.float32,
        # )

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """
        # a workaround to work with gymnasium

        return old_gym_Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32,
        )
        # return spaces.Box(
        #    low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0]),
        #    high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        #    dtype=np.float32,
        # )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.
        Must be implemented in a subclass.
        """

        drone_kinematics = self.drones[0].kinematics  # .position
        # drone_state = self.process_kinematics_to_state(drone_kinematics)
        drone_position = drone_kinematics.position
        drone_velocity = drone_kinematics.velocity

        target_kinematics = self.targets[0].kinematics
        target_position = target_kinematics.position
        target_velocity = target_kinematics.velocity

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

        # TODO adicionar o Survivor Bonus
        # TODO adicionar penalidade por morrer.
        # TODO adicionar bonus por chegar no alvo.\

        max_distance = self.environment_parameters.max_distance

        drone_position = self.drones[0].kinematics.position
        target_position = self.targets[0].kinematics.position
        distance = np.linalg.norm(target_position - drone_position)

        return (max_distance / 10) - 1 * distance

    def _computeDone(self):
        """Computes the current done value(s).
        Must be implemented in a subclass.
        """

        drone_position = self.drones[0].kinematics.position
        drone_velocity = self.drones[0].kinematics.velocity

        target_position = self.targets[0].kinematics.position
        target_velocity = self.targets[0].kinematics.velocity

        distance = np.linalg.norm(target_position - drone_position)

        current = time.time()

        if current - self.RESET_TIME > 20:
            return True

        if (
            distance > self.environment_parameters.max_distance
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
