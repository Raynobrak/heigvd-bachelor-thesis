import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from lidar.LidarSensor import LidarSensor

import pybullet as p

DEFAULT_MAX_EPISODE_DURATION = 10 # secondes
DEFAULT_MAX_VELOCITY = 0.5 # mètres/seconde

DRONE_MODEL = DroneModel.CF2X
PYBULLET_UPDATE_FREQ = 240 # màj physique par seconde
ENV_STEP_FREQ = 48 # appels à env.step() par seconde
DEFAULT_OUTPUT_FOLDER = 'results'
LIDAR_RAYS_COUNT = 102
LIDAR_MAX_DISTANCE = 10

class BaseRLSingleDroneEnv(BaseAviary):
    REWARD_TARGET = np.array([1,1,0.5]) # todo : enlever ce truc en dur

    # Initialise l'environnement
    def __init__(self,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 gui=False,
                 ):
        
        self.max_drone_velocity = max_drone_velocity
        self.max_episode_duration = max_episode_duration

        self.rng = np.random.default_rng(seed=None) # todo : faire autrement

        super().__init__(drone_model=DRONE_MODEL,
                         num_drones=1,
                         neighbourhood_radius=0,
                         initial_xyzs=initial_xyz_position,
                         initial_rpys=initial_rpy_attitude,
                         pyb_freq=PYBULLET_UPDATE_FREQ,
                         ctrl_freq=ENV_STEP_FREQ,
                         gui=gui,
                         record=False,
                         obstacles=True,
                         user_debug_gui=False,
                         output_folder=DEFAULT_OUTPUT_FOLDER
                         )
        
        self.lidar_sensor = LidarSensor(
            freq=20,
            pybullet_client_id=self.getPyBulletClient(),
            pybullet_drone_id=self.getDroneIds()[0],
            nb_angles=LIDAR_RAYS_COUNT - 2,
            max_distance=LIDAR_MAX_DISTANCE,
            show_debug_rays=True
        )

        assert(LIDAR_RAYS_COUNT == self.lidar_sensor.rays_count())

        self.terminated = False # todo : gérer la termination autrement

        self.pid_controller = DSLPIDControl(drone_model=self.DRONE_MODEL) # todo : constantes + mettre ça ailleurs peut-être

        self.obstacles_ids = []
    
    # Méthode appelée par la classe mère (BaseAviary) au moment de reset() l'environnement
    # Ajoute les obstacles dans l'environnement
    def _addObstacles(self):
        half_extents = [1, 1, 0.2]
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.CLIENT
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            basePosition=[-self.rng.uniform(1,5), 0, half_extents[2] * 2],
            physicsClientId=self.CLIENT
        )    

    # retourne la position réelle du drone dans la simulation (ground-truth)
    def get_real_drone_pos(self):
        return self.pos[0,:] 
    
    # retourne la position estimée du drone dans la simulation
    # par défaut, c'est la position réelle. en cas d'implémentation d'un SLAM, surcharger cette méthode
    def get_estimated_drone_pos(self):
        return self.get_real_drone_pos() #todo : émuler slam
    
    # retourne la vitesse réelle (vecteur 3D) du drone dans la simulation (ground-truth)
    def get_real_drone_velocity(self):
        return self.vel[0,:]

    
    # retourne la vitesse (vecteur 3D) estimée du drone dans la simulation
    # par défaut, c'est la vitesse réelle. en cas d'implémentation d'un SLAM, surcharger cette méthode
    def get_estimated_drone_velocity(self):
        return self.get_real_drone_velocity() # todo : émuler slam
    
        # retourne l'orientation réelle du drone en RPY (roll, pitch, yaw)
    def get_real_drone_attitude(self):
        return self.rpy[0,:]
    
    # retourne une estimation de l'attitude du drone en RPY (roll, pitch, yaw)
    def get_estimated_drone_attitude(self):
        return self.get_real_drone_attitude() # todo : émuler slam

    # retourne le temps écoulé depuis le début de l'épisode
    def get_elapsed_time(self):
        return self.step_counter / self.PYB_FREQ

    def _actionSpace(self):
        # l'espace d'action est un vecteur en 3 dimensions censé représenter la vitesse cible
        # cette information est ensuite passée au contrôleur PID qui se chargera d'ajuster
        # les RPMs des quatres moteurs pour atteindre la vélocité désirée.
        return spaces.Box(
            low=-self.max_drone_velocity,
            high=self.max_drone_velocity,
            shape=(3,),
            dtype=np.float32,
        )
    
    def _observationSpace(self):
        # l'espace d'observation est un vecteur représentant l'état actuel du drone et de son environment
        # en l'occurence, il s'agit de sa position, sa vitesse et son orientation ainsi que des N scans du capteur LiDAR
        low = np.concatenate([
            np.full(9, -np.inf),
            np.zeros(LIDAR_RAYS_COUNT)
        ])
        high = np.concatenate([
            np.full(9, np.inf),
            np.full(LIDAR_RAYS_COUNT, LIDAR_MAX_DISTANCE)
        ])

        return spaces.Box(low=low, high=high, dtype=np.float32)

    # Retourne une observation de l'environnement (état du drone et de ses capteurs)
    def _computeObs(self):
        data = self.lidar_sensor.update(1. / self.CTRL_FREQ)

        observation = np.zeros(self._observationSpace().shape)
        observation[0:3] = self.get_estimated_drone_pos()
        observation[3:6] = self.get_estimated_drone_velocity()
        observation[6:9] = self.get_estimated_drone_attitude()

        observation[9:9 + LIDAR_RAYS_COUNT] = self.lidar_sensor.read_distances()

        return observation

    # Méthode appellée automatiquement par la classe parente.
    # Convertir l'action (voir _actionSpace()) en 4 valeurs : RPM des quatres moteurs.
    # Dans notre cas, l'action (vecteur vitesse cible) est convertie en RPM grâce à l'implémentation
    # de DSLPIDControl, un contrôleur PID implémenté par gym-pybullet-drones et qui provient de UTIAS DSL (Dynamic Systems Lab)
    def _preprocessAction(self, action):
        #todo refactor
        obs = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        state = obs[0]
        target_pos = self.get_real_drone_pos() # todo : position réelle ou estimée ?
        target_rpy = self.INIT_RPYS[0,:]
        target_vel = action.reshape(-1)

        # todo : normaliser la vitesse pour que ça rentre dans self.max_drone_velocity
        
        target_rpms = self.pid_controller.computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                  state=state,
                                                                  target_pos=target_pos,
                                                                  target_rpy=target_rpy, # on reste horizontal
                                                                  target_vel=target_vel)
        
        return np.array(np.clip(target_rpms[0], 0, self.MAX_RPM))
    
    # Retourne la distance euclidienne entre le drone et le point cible
    def distance_to_target(self):
        current_pos = self.get_estimated_drone_pos()
        distance = np.linalg.norm(self.REWARD_TARGET - current_pos)
        return distance

    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        distance = self.distance_to_target()

        error_radius = 0.01
        #reward = (50/(distance + error_radius)) / self.CTRL_FREQ
        # todo : reward proportionnel à la fréquence de controle de la simulation (pour éviter que ça soit déséquilibré si on change la fréquence)

        reward = np.linalg.norm(self.get_estimated_drone_pos() - self.INIT_XYZS[0,:]) / self.CTRL_FREQ

        # todo : réactiver les collisions
        if p.getContactPoints(bodyA=self.DRONE_IDS[0]):
            print('collision')
            reward -= 5000
            self.terminated = True
        
        return reward
    
    # Retourne True si l'épisode doit être considéré comme étant terminé.
    # Suite au crash du drone (collision) par exemple ou au dépassement du temps de vol.
    def _computeTerminated(self):
        if self.get_elapsed_time() > self.max_episode_duration:
            self.terminated = True

        # si on est assez proche de la cible, l'épisode est terminé
        # todo : voir pour remettre cette condition
        #if self.distance_to_target() < 0.1:
        #    self.terminated = True
        #    return True

        return self.terminated    
    

    def _computeTruncated(self):
        # todo : faire correctement. terminated = crash ou réussite, trunacted = dépassement du temps ou sortie des limites de la simulation
        return False
    
    def _computeInfo(self):
        return {'info':None}
