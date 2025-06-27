import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

import pybullet as p

# TODO : adapter

class ReinforcementLearningEnv(BaseAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

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
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        self.terminated = False

        self.pid_controller = DSLPIDControl(drone_model=DroneModel("cf2x")) # todo : constantes + mettre ça ailleurs peut-être

        
        # todo : générer les obstacles différemment
        #p.connect()
        # Taille (20, 20, 50) → demi-extents = moitié de chaque côté
        half_extents = [0.5, 0.5, 1]

        # Créer la forme de collision (box)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )

        # Créer le corps rigide (statique ici)
        self.obstacle_id = p.createMultiBody(
            baseMass=0,  # 0 = statique
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=[2, 2, half_extents[2]]  # centre de la boîte à 25 pour que sa base soit à z=0
        )

    ################################################################################

    def _actionSpace(self):
        # the action space is a 3 dimensional vector representing the target velocity
        return spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32,
        )
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the enfvironment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """

        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self, action):


        """     
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.
        """

        obs = self._computeObs()
        state = obs[0]
        target_pos = obs[0,0:3].reshape(-1)
        target_rpy = self.INIT_RPYS[0,:]
        target_vel = action.reshape(-1)

        print(state.shape)
        print(target_pos.shape)
        print(target_rpy.shape)
        print(target_vel.shape)
        
        target_rpms = self.pid_controller.computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                  state=state,
                                                                  target_pos=target_pos,
                                                                  target_rpy=target_rpy, # on reste horizontal
                                                                  target_vel=target_vel)
        
        return np.array(np.clip(target_rpms[0], 0, self.MAX_RPM))
        return np.array([np.clip(target_rpms[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):

        TARGET = np.array([2,2,1])

        current_pos = self.pos[0,:]
        distance = np.linalg.norm(TARGET - current_pos)

        error_radius = 0.01
        reward = 1/(distance + error_radius) - 0.5

        if p.getContactPoints(bodyA=self.DRONE_IDS[0]):
            reward -= 5000
            self.terminated = True

        # todo : reward proportionnel à la fréquence de controle de la simulation (pour éviter que ça soit déséquilibré si on change la fréquence)
        
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        return self.terminated    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
