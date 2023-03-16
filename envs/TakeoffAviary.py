import numpy as np
import pybullet as p
import math

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviarySimplified import ActionType, ObservationType, BaseSingleAgentAviary
from random import random, choice
from gym import spaces

'''
O design do projeto está orientado ao funcional. Infelizmente, é necessário ter esse objeto para a criação do ambiente.

Devo então:
1. Evitar ao máximo variáveis globais
2. Diminuir ao máximo o número de ações (funções impuras que causam efeitos colaterais).
'''


class TakeoffAviary(BaseSingleAgentAviary):
    """Single agent RL problem: take-off."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL
                 ):
        """Initialization of a single agent RL environment.
        Using the generic single agent RL superclass.
        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        """
        NUM_DRONES = 1
        initial_xyzs = np.zeros((NUM_DRONES, 3))
        initial_xyzs[0] = self.gen_random_position()


        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        self._addObstacles()

        self.M_Cube = p.getDynamicsInfo(self.cube, -1, physicsClientId=self.CLIENT)[0]

        self.cube_pos = np.zeros(3)
        self.cube_quat = np.zeros(4)
        self.cube_rpy = np.zeros(3)
        self.cube_vel = np.zeros(3)
        self.cube_ang_v = np.zeros(3)

    ################################################################################

    def gen_random_position():
        x = choice([1, -1])
        y = choice([1, -1])
        z = 1

        return [x * random() / 2, y * random() / 2, z * random() / 2 + 0.2]


    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """

        bonus = 0
        penalty = 0

        c_pos = self.cube_obs[:3]
        d_pos = self.obs[:3]
        vect = c_pos - d_pos
        dist = abs(np.linalg.norm(vect))

        penalty = dist

        if self.pos[0, 2] < 0.3:
            penalty += 10

        return bonus - penalty

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.
        Returns
        -------
        bool
            Whether the current episode is done.
        """

        c_pos = self.cube_obs[:3]
        d_pos = self.obs[:3]
        vect = c_pos - d_pos
        dist = abs(np.linalg.norm(vect))

        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
        # Alternative done condition, see PR #32
        # if (self.step_counter/self.SIM_FREQ > (self.EPISODE_LEN_SEC)) or ((self._getDroneStateVector(0))[2] < 0.05):
            return True

        elif dist < 5 * 1e-3:

            return True
        else:

            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).
        Unused.
        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
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

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.
        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


    def _addObstacles(self):
        """Add obstacles to the environment.
        These obstacles are loaded from standard URDF files included in Bullet.
        """

        #random = np.random
        k1 = choice([1, -1])
        k2 = choice([1, -1])
        k3 = 1#choice([1, -1])

        self.cube_pos = [k1 * random() / 2, k2 * random() / 2, k3 * random() / 2 + 0.2]

        #self.cube_pos = np.array([0, 0, 0.5])
        self.cube = p.loadURDF("cube_small.urdf",
                   self.cube_pos,
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )

        #p.resetBaseVelocity(self.cube, [random()/2,random()/2,random()/2], [0, 0, 1], physicsClientId=self.CLIENT)





    def _dynamics(self, rpm, nth_drone):
        super()._dynamics(rpm, nth_drone)

    def cube_step(self):
        p.applyExternalForce(self.cube, -1, forceObj=[0, 0, self.M_Cube * self.G], posObj=[
                             0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

        self.cube_vel = np.zeros(3)

        p.resetBaseVelocity(self.cube, self.cube_vel,
                            [0, 0, 0], physicsClientId=self.CLIENT)


    def step(self, action):
        self.cube_step()
        return super().step(action)

    def _computeObs(self):
        """Returns the current observation of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.
        """

        cube_vel = self.cube_vel
        cube_motors_speed = [0,0,0,0]


        cube_state = np.hstack([self.cube_pos, self.cube_quat, self.cube_rpy,
                           self.cube_vel, self.cube_ang_v, cube_motors_speed])

        cube_state = cube_state.reshape(20,)
        cube_obs = self._clipAndNormalizeState(cube_state)
        self.cube_obs = cube_obs

        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        self.obs = obs

        d_pos = obs[0:3]
        d_ang_pos = obs[7:10]
        d_vel = obs[10:13]
        d_ang_vel = obs[13:16]

        c_pos = cube_obs[0:3]
        c_vel = cube_obs[10:13]

        #vect = self.cube_pos - self.pos[0, :]
        vect = c_pos - d_pos
        dist = abs(np.linalg.norm(vect))


        #vect_unitary = vect/dist if dist != 0 else vect
        vect_unitary = vect
        dist = dist/10 if dist < 10 or dist > -10 else 1

        #ret = np.hstack([d_pos, d_vel, c_pos, vect_unitary, dist]).reshape(13,)
        ret = np.hstack([d_pos, d_vel, c_pos, c_vel, vect_unitary, dist]).reshape(16,)
        #print(ret)
        return ret.astype('float32')

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """

        #position, angular_position, velocity, angular_velocity

        #return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, -1, -1, 0]),
        #                  high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1, 1, 1, 1]),
        #                  dtype=np.float32
        #                  )

        return spaces.Box(low=np.array( [-1,-1, 0,-1,-1,-1,             -1,-1, 0,-1,-1,-1,   -1,-1,-1,   0]),
                          high=np.array([ 1, 1, 1,1, 1, 1,               1, 1, 1, 1, 1, 1,    1, 1, 1,   1]),
                          dtype=np.float32
                          )


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
        NUM_DRONES = 1
        initial_xyzs = np.zeros((NUM_DRONES, 3))

        k1 = choice([1, -1])
        k2 = choice([1, -1])
        k3 = 1#choice([1, -1])
        initial_xyzs[0] = [k1 * random()/1.5, k2 * random()/1.5, k3 * random()/1.5 + 0.2]



        self.INIT_XYZS = initial_xyzs

        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()
