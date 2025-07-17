import time
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from scipy.spatial.transform import Rotation as R

from lidar.LidarSensor import LidarSensor
from mapping.map import Map

from .Action import *

import pybullet as p

DRONE_MODEL = DroneModel.CF2X

DEFAULT_MAX_EPISODE_DURATION = 10 # secondes
DEFAULT_MAX_VELOCITY = 0.6 # mètres/seconde

DEFAULT_PYBULLET_PHYSICS_FREQ = 240 # màj physique par seconde, même valeur que l'environnement de base BaseAviary (provenant du projet gym-pybullet-drones original)
DEFAULT_ENV_STEP_FREQ = 60 # appels à env.step() par seconde
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_LIDAR_RAYS_COUNT = 14
DEFAULT_LIDAR_MAX_DISTANCE = 12
DEFAULT_LIDAR_FREQUENCY = DEFAULT_ENV_STEP_FREQ
DEFAULT_ENABLE_LIDAR_RAYS = False
DEFAULT_DRONE_LATERAL_SPEED_MULTIPLIER = 0.5

class BaseRLSingleDroneEnv(BaseAviary):
    REWARD_TARGET = np.array([1,1,0.5]) # todo : enlever ce truc en dur

    # Initialise l'environnement
    def __init__(self,
                 pybullet_physics_freq=DEFAULT_PYBULLET_PHYSICS_FREQ,
                 env_step_freq=DEFAULT_ENV_STEP_FREQ,
                 lidar_rays_count=DEFAULT_LIDAR_RAYS_COUNT,
                 lidar_max_distance=DEFAULT_LIDAR_MAX_DISTANCE,
                 lidar_freq=DEFAULT_LIDAR_FREQUENCY,
                 enable_lidar_rays_debug= DEFAULT_ENABLE_LIDAR_RAYS,
                 enable_mapping=False,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 drone_lateral_speed_multiplier = DEFAULT_DRONE_LATERAL_SPEED_MULTIPLIER,
                 gui=False,
                 ):
        
        self.lidar_rays_count = lidar_rays_count
        self.lidar_max_distance = lidar_max_distance
        self.lidar_freq = lidar_freq
        self.enable_lidar_rays_debug = enable_lidar_rays_debug
        self.enable_mapping = enable_mapping
        
        self.max_drone_velocity = max_drone_velocity
        self.max_episode_duration = max_episode_duration

        self.drone_lateral_speed_multiplier = drone_lateral_speed_multiplier

        self.rng = np.random.default_rng(seed=None) # todo : faire autrement
        super().__init__(drone_model=DRONE_MODEL,
                         num_drones=1,
                         neighbourhood_radius=0,
                         initial_xyzs=initial_xyz_position,
                         initial_rpys=initial_rpy_attitude,
                         pyb_freq=pybullet_physics_freq,
                         ctrl_freq=env_step_freq,
                         gui=gui,
                         record=False,
                         obstacles=True,
                         user_debug_gui=False,
                         output_folder=DEFAULT_OUTPUT_FOLDER
                         )
                
        self.lidar_sensor = self.build_lidar_sensor()
        assert(self.lidar_rays_count == self.lidar_sensor.rays_count())

        self.custom_reset()

    def build_lidar_sensor(self):
        return LidarSensor(
            freq=self.lidar_freq,
            pybullet_client_id=self.getPyBulletClient(),
            pybullet_drone_id=self.getDroneIds()[0],
            nb_angles=self.lidar_rays_count - 2,
            max_distance=self.lidar_max_distance,
            show_debug_rays=self.enable_lidar_rays_debug
        )

    def specific_reset(self):
        print('not implemented')

    def custom_reset(self):
        self.pid_controller = DSLPIDControl(drone_model=self.DRONE_MODEL) # todo : constantes + mettre ça ailleurs peut-être
        self.time_elapsed_text_id = None
        self.specific_reset()

    def reset(self, seed : int = None, options : dict = None):
        self.custom_reset()
        return super().reset(seed, options)
    
    # Méthode appelée par la classe mère (BaseAviary) au moment de reset() l'environnement
    # Ajoute les obstacles dans l'environnement
    def _addObstacles(self):
        pass

    def add_fixed_obstacle(self, center_pos, size, rgba_color=[1,0,0,0.3], rotation_rpy=[0,0,0]):
        center_pos = np.array(center_pos)
        size = np.array(size)

        half_size = size / 2
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_size,
            physicsClientId=self.CLIENT
        )

        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size, rgbaColor=rgba_color)

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos,
            physicsClientId=self.CLIENT,
            baseOrientation=p.getQuaternionFromEuler(rotation_rpy)
        )

        # todo : voir pour utiliser une texture
        #tex_id = p.loadTexture(r'C:\Users\lcsch\OneDrive - HESSO\Semestre6\TB\heigvd-bachelor-thesis\drone-rl-environment\src\drone_rl_environment\texture.png')
        #p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)

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
        return self.step_counter * self.PYB_TIMESTEP

    def _actionSpace(self):
        return spaces.Discrete(int(Action.ACTIONS_COUNT)) # stop, avant, haut, bas, gauche, droite, rotation gauche, rotation droite
    
    def _observationSpace(self):
        # espace d'observation lidar-only
        low = np.concatenate([
            np.zeros(self.lidar_rays_count)
        ])
        high = np.concatenate([
            np.full(self.lidar_rays_count, 1)
        ])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # Retourne une observation de l'environnement (état du drone et de ses capteurs)
    def _computeObs(self):
        self.lidar_sensor.update(1. / self.CTRL_FREQ)

        # todo : mettre l'affichage de debug ailleurs
        if self.GUI:
            if self.time_elapsed_text_id is not None:
                p.removeUserDebugItem(self.time_elapsed_text_id)
            self.time_elapsed_text_id = p.addUserDebugText(str(round(self.get_elapsed_time(),1)) + ' / ' + str(round(self.max_episode_duration,1)) + ' seconds.', self.get_real_drone_pos(), textColorRGB=[0, 1, 0], textSize=1.5)

        observation = np.zeros(self._observationSpace().shape)
        observation[0:self.lidar_rays_count] = np.array(self.lidar_sensor.read_distances()) / self.lidar_max_distance

        return observation
    
    def reset_map(self):
        if self.enable_mapping: 
            size = 20 # todo : constantes/paramètres
            origin = size / 2
            res = 50
            print('map resetted')
            self.map = Map(x_size=size, y_size=size, z_size=size, origin_offset=np.array([origin,origin,origin]), resolution_voxels_per_unit=res) # todo : construire que si activé
        else:
            raise('error : could not create map; mapping is disabled.')
    
    def update_map(self):
        if self.is_done():
            print('uuuuh wtf')
        if self.enable_mapping:
            self.map.add_scan(local_scan_points=self.lidar_sensor.read_local_points(), sensor_position=self.get_real_drone_pos(), max_distance=self.lidar_max_distance)
        else:
            raise('error : could not update map; mapping is disabled.')
        
    def is_done(self):
        return self._computeTerminated() or self._computeTruncated()
    
    def rotate_vector_by_rpy(self, pos, rpy):
        orientation_quat = p.getQuaternionFromEuler(rpy)
        rot_matrix = R.from_quat(orientation_quat).as_matrix()
        return rot_matrix @ np.array(pos)

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
        
        #target_vel = action.reshape(-1)

        # todo : normaliser la vitesse pour que ça rentre dans self.max_drone_velocity

        target_pos = self.get_real_drone_pos()

        target_rpy = self.get_real_drone_attitude()
        target_dir = np.array(action_to_direction(action))

        # si le drone va en avant, la vélocité est maximale. si le drone va sur les côtés (DRIFT_LEFT ou DRIFT_RIGHT), cette vitesse est diminuée (par self.drone_lateral_speed_multiplier)
        speed = self.max_drone_velocity if action == Action.FORWARD else self.max_drone_velocity * self.drone_lateral_speed_multiplier
        target_vel = target_dir * speed

        target_vel = self.rotate_vector_by_rpy(target_vel, target_rpy)
        target_rpy_rates = [0,0,0]
        if action == Action.ROTATE_LEFT:
            target_rpy_rates = [0,0,np.deg2rad(360 / 3)] # vitesse de rotation : 3 tour par seconde # todo : constante pour vitesse de rotation
        elif action == Action.ROTATE_RIGHT:
            target_rpy_rates = [0,0,-np.deg2rad(360 / 3)]

        target_rpms = self.pid_controller.computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                  state=state,
                                                                  target_pos=target_pos,
                                                                  target_rpy=target_rpy,
                                                                  target_vel=target_vel,
                                                                  target_rpy_rates=target_rpy_rates)
        
        return np.array(np.clip(target_rpms[0], 0, self.MAX_RPM))
    
    def save_map(self, filename):
        if not self.enable_mapping:
            raise('error : could not save map; mapping is disabled.')
        else:
            self.map.save_2D_map_to_file(filename)
    
    # Retourne la distance euclidienne entre le drone et le point cible
    def distance_to_target(self):
        current_pos = self.get_estimated_drone_pos()
        distance = np.linalg.norm(self.REWARD_TARGET - current_pos)
        return distance
    
    # retourne True si le drone est en contact avec un obstacle, sinon False
    def check_for_collisions(self):
        if p.getContactPoints(bodyA=self.DRONE_IDS[0]):
            return True
        return False

    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        return 0
    
    # Retourne True si l'épisode doit être considéré comme étant terminé.
    def _computeTerminated(self):
        return self.check_for_collisions()    

    def _computeTruncated(self):
        return self.get_elapsed_time() >= self.max_episode_duration
    
    def _computeInfo(self):
        return {'info':None}
