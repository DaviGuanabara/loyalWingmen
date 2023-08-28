import time

# import curses
import random
import math

from pynput.keyboard import Key, KeyCode
from collections import defaultdict

import numpy as np
import pybullet as p
import pybullet_data


import gymnasium as gym
from gymnasium import spaces, Env


from modules.factories.drone_factory import DroneFactory
from modules.factories.loiteringmunition_factory import (
    LoiteringMunitionFactory,
    LoiteringMunition,
)
from modules.factories.loyalwingman_factory import LoyalWingmanFactory, LoyalWingman

from modules.dataclasses.dataclasses import EnvironmentParameters
from modules.models.lidar import CoordinateConverter
from typing import List
from modules.models.lidar import Channels

from pathlib import Path
import os
from typing import Tuple
import math

class DroneChaseEnvLevel1(Env):
    """
    This class aims to demonstrate a environment with one Loyal Wingmen and one Loitering Munition,
    in a simplest way possible.
    It was developed for Stable Baselines 3 2.0.0 or higher.
    Gym Pybullet Drones was an inspiration for this project. For more: https://github.com/utiasDSL/gym-pybullet-drones

    Finally, I tried to do my best inside of my time constraint. Then, sorry for messy code.
    """

    def __init__(
        self,
        simulation_frequency: int = 240,
        rl_frequency: int = 30,
        gravity: float = 9.81,
        GUI: bool = False,
        debug: bool = False,
        
    ):
    
        if debug:
            print("DroneChaseEnvLevel1::__init__")
            print("simulation_frequency: ", simulation_frequency)
            print("rl_frequency: ", rl_frequency)
            print("GUI: ", GUI)
            
        self.setup_pybullet(simulation_frequency, rl_frequency, gravity, GUI) 
        
        self.drone_id = self.create_drone([1, 1, 0])
        self.aggregate_physics_steps = int(simulation_frequency / rl_frequency)   
        self.MAX_DISTANCE = 10
        self.MAX_VELOCITY = .1
        self.MAX_TIME = 5#10
        self.observation_space = self._observationSpace()
        self.action_space = self._actionSpace()
            
        
    #### Setups ##########################################    
        
        
    
    def setup_pybullet(self, simulation_frequency, rl_frequency, gravity, GUI):
        self.simulation_frequency = simulation_frequency
        self.rl_frequency = rl_frequency    
        
        if GUI:
            self.client_id = self.setup_pybulley_GUI()
            
        else:
            self.client_id = self.setup_pybullet_DIRECT()
            
        p.setGravity(
            0,
            0,
            0,#-gravity,
            physicsClientId=self.client_id,
        )
        
        p.setRealTimeSimulation(
            0, 
            physicsClientId=self.client_id
        )  # No Realtime Sync
        
        p.setTimeStep(
            1 / simulation_frequency,
            physicsClientId=self.client_id,
        )
    
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
            

    def create_drone(self, position) -> int:
        urdf_name = "cf2x"  + ".urdf"
        base_path = str(Path(os.getcwd()).parent.absolute())
        
        urdf_file_path = base_path + "\\" + "assets\\" + urdf_name  # "cf2x.urdf"

        id = p.loadURDF(
            fileName=urdf_file_path,
            basePosition=position,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        
        return id
        
    
    
    
    
    
    #### Housekeeping ##########################################
    
    
    def reset(self, seed=1)-> Tuple[np.ndarray, dict]:
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        
        self.RESET_TIME = time.time() 
        
        p.resetBasePositionAndOrientation(
            
            bodyUniqueId=self.drone_id,
            posObj=[1, 1, 0],
            ornObj=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        
        p.resetBaseVelocity(
                self.drone_id,
                [0, 0, 0],
                np.array([0, 0, 0]),
                physicsClientId=self.client_id,
            )

        observation = self._compute_observation()
        info = self._compute_info()
        return (observation, info)
    
    
    def step(self, rl_action: np.ndarray):
        
        velocity = rl_action
        velocity = np.concatenate([rl_action, [0]])

        p.resetBaseVelocity(
                self.drone_id,
                velocity,
                np.array([0, 0, 0]),
                physicsClientId=self.client_id,
            )
        for _ in range(self.aggregate_physics_steps):
            
            p.stepSimulation()

        observation = self._compute_observation()
        reward = self._compute_reward()
        terminated = self._compute_done()
        info = self._compute_info()

        return observation, reward, terminated, False, info
    
    def _compute_observation(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client_id,
            )
    
        normalized = self._normalize_position(np.array(position, dtype=np.float32))
        
        return normalized[0:2]
    
    def _compute_reward(self) -> float:
        position, _ = p.getBasePositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client_id,
            )
        
        
        #norm_distance: float = float(np.linalg.norm(position))
        
        #if norm_distance < .2:
        #    return 1_000_000
        
        return -10 * float(np.linalg.norm(position)) **2
    
    def _compute_done(self):
        current = time.time()
        
        if current - self.RESET_TIME > self.MAX_TIME: 
            return True

        return False 
    
    def _compute_info(self):
        return {}
    
    def _observationSpace(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
    
    def _actionSpace(self):
        return spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)
        
        
    #####################################################################################################
    # Normalization
    #####################################################################################################

    def _normalizeVelocity(self, velocity: np.ndarray) -> np.ndarray:
        
        normalized_velocity = (np.clip(velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY) /
                            np.full_like(velocity, self.MAX_VELOCITY))
        return normalized_velocity

    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        
        normalized = (np.clip(position, -self.MAX_DISTANCE, self.MAX_DISTANCE) /
                            np.full_like(position, self.MAX_DISTANCE))
        
        return normalized
    
    def get_keymap(self):
        keycode = KeyCode()
        default_action = [0.0, 0.0]

        key_map = defaultdict(lambda: default_action)
        key_map.update({
            Key.up: [0, .1],
            Key.down: [0, -.1],
            Key.left: [-.1, 0],
            Key.right: [.1, 0],
            #keycode.from_char("w"): [0, 0, 1.0, .1],
            #keycode.from_char("s"): [0, 0, -1.0, .1],
        })
        
        return key_map  