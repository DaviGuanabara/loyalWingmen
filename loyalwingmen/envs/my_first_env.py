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


class MyFirstEnv(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {'render.modes': ['human']}

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 GUI: bool = False
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
        self.G = 9.8


        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.PHYSICS = physics
        self.URDF = self.DRONE_MODEL.value + ".urdf"
      
  

        self.step_counter = 0
        #### Connect to PyBullet ###################################
       
        #### With debug GUI ########################################
        # p.connect(p.GUI, options="--opengl2")

        if GUI:

            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                            cameraYaw=-30,
                                            cameraPitch=-30,
                                            cameraTargetPosition=[0, 0, 0],
                                            physicsClientId=self.CLIENT
                                            )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])

        else:
            self.CLIENT = p.connect(p.DIRECT)
       
        
            #### Without debug GUI #####################################
            #self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            # Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            
        #### Set initial poses #####################################
       
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        #self._housekeeping()
        #### Update and store the drones kinematic information #####
        #self._updateAndStoreKinematicInformation()

        

    ################################################################################

    #def reset(self):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
    #    p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
    #    self._housekeeping()
        #### Update and store the drones kinematic information #####
        #self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        #self._startVideoRecording()
        #### Return the initial observation ########################
    #    return self._computeObs()
    
    def reset(self):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        initial_xyzs = np.zeros(3)
        


        #self._housekeeping()
        #### Update and store the drones kinematic information #####
        #self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        #self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()

    ################################################################################

    def step(self,
             action
             ):
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

        #### Read the GUI's input parameters #######################

        #todo corrigir
        #clipped_action = np.reshape(
        #    self._preprocessAction(action))
        #### Repeat for as many as the aggregate physics steps #####
        #for _ in range(self.AGGR_PHY_STEPS):
            
            #Physics.PYB:
            #self._physics(clipped_action[i, :], i)
      
            #### PyBullet computes the new state, unless Physics.DYN ###
            #if self.PHYSICS != Physics.DYN:
            #    p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            #self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        #self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################

        self.step_counter += 1

        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        return obs, reward, done, info

    ################################################################################


    def close(self):
        """Terminates the environment.
        """
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
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
  
        # print("load drones position:", self.INIT_XYZS[0,:])
        #self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
        #                                      self.INIT_XYZS[i, :],
        #                                      p.getQuaternionFromEuler(
        #                                          self.INIT_RPYS[i, :]),
        #                                      flags=p.URDF_USE_INERTIA_FROM_FILE,
        #                                      physicsClientId=self.CLIENT
        #                                      ) for i in range(self.NUM_DRONES)])
        #if self.OBSTACLES:
        #    self._addObstacles()

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.
        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """
        #for i in range(self.NUM_DRONES):
        #    self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(
        #        self.DRONE_IDS[i], physicsClientId=self.CLIENT)
        #    self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
        #    self.vel[i], self.ang_v[i] = p.getBaseVelocity(
        #        self.DRONE_IDS[i], physicsClientId=self.CLIENT)

    ################################################################################



    def _getDroneStateVector(self,
                             nth_drone
                             ):
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
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################



  

    ################################################################################

    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )




    ################################################################################

    #def _actionSpace(self):
    #    """Returns the action space of the environment.
    #    Must be implemented in a subclass.
    #    """
    #    #raise NotImplementedError
    #    return []
    
    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.
        """
        # if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        #    size = 4
        # else:
        #    print("[ERROR] in BaseSingleAgentAviary._actionSpace()")
        #    exit()

        # return spaces.Box(low=-1*np.ones(size),
        # return spaces.Box(low=np.zeros(size),  # Alternative action space, see PR #32
        #                  high=np.ones(size),
        #                  dtype=np.float32
        #                  )

        return spaces.Box(low=np.array([-1, -1, -1, 0.4]),  # Alternative action space, see PR #32
                          high=np.array([1,  1,  1,   1]),
                          dtype=np.float32
                          )

    ################################################################################

    #def _observationSpace(self):
    #    """Returns the observation space of the environment.
    #    Must be implemented in a subclass.
    #    """
    #    #raise NotImplementedError
    #    return []
    
    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """

        return spaces.Box(low=np.array([-1, -1, 0, -1, -1, -1,             -1, -1, 0, -1, -1, -1,   -1, -1, -1,   0]),
                          high=np.array([1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,    1, 1, 1,   1]),
                          dtype=np.float32
                          )


    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.
        Must be implemented in a subclass.
        """
        #raise NotImplementedError
        return np.zeros(16).astype('float32')

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.
        Must be implemented in a subclass.
        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.
        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).
        Must be implemented in a subclass.
        """
        #raise NotImplementedError
        return 0

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).
        Must be implemented in a subclass.
        """
        #raise NotImplementedError

        if self.step_counter % 100 == 0:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).
        Must be implemented in a subclass.
        """
        #raise NotImplementedError
        return {}
