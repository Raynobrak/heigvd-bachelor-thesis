import time
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from scipy.spatial.transform import Rotation as R

from drone_rl_environment.lidar.LidarSensor import LidarSensor
from drone_rl_environment.mapping.map import Map

from .Action import *

import pybullet as p

DRONE_MODEL = DroneModel.CF2X

DEFAULT_MAX_EPISODE_DURATION = 10 # secondes
DEFAULT_MAX_VELOCITY = 0.6 # mètres/seconde

# 240 60 5 old
DEFAULT_PYBULLET_PHYSICS_FREQ = 240 # màj physique par seconde, même valeur que l'environnement de base BaseAviary (provenant du projet gym-pybullet-drones original)
DEFAULT_PID_CONTROLLER_FREQ = 60 # fréquence de contrôle du contrôlleur PID
DEFAULT_ACTION_FREQ = 5 # fréquence d'appel à step(), fréquence d'action de la simulation

DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_LIDAR_RAYS_COUNT = 14
DEFAULT_LIDAR_MAX_DISTANCE = 12
DEFAULT_LIDAR_FREQUENCY = DEFAULT_PID_CONTROLLER_FREQ
DEFAULT_ENABLE_LIDAR_RAYS = False
DEFAULT_DRONE_UP_AND_DOWN_SPEED_MULTIPLIER = 0.5
DEFAULT_DRONE_MAX_ROTATION_RATE = np.deg2rad(360 / 3) # vitesse de rotation (yaw = lacet) exprimée en radians/seconde

DRONES_COUNT = 1

DEFAULT_PHYSICS = Physics.PYB

MAP_WIDTH_METERS = 20 # largeur de la zone mappable (voir Map.py)
MAP_HEIGHT_METERS = 5 # hauteur de la zone mappable (voir Map.py)
MAP_RESOLUTION = 10 # sous-division de la zone mappable. 10 -> chaque mètre cube est divisée en blocs de 10x10x10cm

class BaseRLSingleDroneEnv(BaseAviary):
    # Initialise l'environnement
    def __init__(self,
                 pybullet_physics_freq=DEFAULT_PYBULLET_PHYSICS_FREQ,
                 pid_controller_freq=DEFAULT_PID_CONTROLLER_FREQ,
                 action_freq=DEFAULT_ACTION_FREQ,
                 lidar_rays_count=DEFAULT_LIDAR_RAYS_COUNT,
                 lidar_max_distance=DEFAULT_LIDAR_MAX_DISTANCE,
                 lidar_freq=DEFAULT_LIDAR_FREQUENCY,
                 enable_lidar_rays_debug= DEFAULT_ENABLE_LIDAR_RAYS,
                 enable_mapping=False,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 drone_up_and_down_speed_multiplier = DEFAULT_DRONE_UP_AND_DOWN_SPEED_MULTIPLIER,
                 gui=False,
                 ):
        
        self.lidar_rays_count = lidar_rays_count
        self.lidar_max_distance = lidar_max_distance
        self.lidar_freq = lidar_freq
        self.enable_lidar_rays_debug = enable_lidar_rays_debug
        self.enable_mapping = enable_mapping
        
        self.max_drone_velocity = max_drone_velocity
        self.max_episode_duration = max_episode_duration

        self.drone_up_and_down_speed_multiplier = drone_up_and_down_speed_multiplier

        self.action_freq = action_freq

        self.rng = np.random.default_rng(seed=None)

        super().__init__(drone_model=DRONE_MODEL,
                         num_drones=DRONES_COUNT,
                         neighbourhood_radius=0,
                         initial_xyzs=initial_xyz_position,
                         initial_rpys=initial_rpy_attitude,
                         physics=DEFAULT_PHYSICS,
                         pyb_freq=pybullet_physics_freq,
                         ctrl_freq=pid_controller_freq,
                         gui=gui,
                         record=False,
                         obstacles=True,
                         user_debug_gui=False,
                         output_folder=DEFAULT_OUTPUT_FOLDER
                         )
                

        self.custom_reset()
        assert(self.lidar_rays_count == self.lidar_sensor.rays_count())


    def build_lidar_sensor(self):
        return LidarSensor(
            freq=self.lidar_freq,
            pybullet_client_id=self.getPyBulletClient(),
            pybullet_drone_id=self.getDroneIds()[0],
            nb_angles=self.lidar_rays_count - 2,
            max_distance=self.lidar_max_distance,
            show_debug_rays=self.enable_lidar_rays_debug
        )

    # réinitialisation spécifique appelée à la fin de custom_reset(), à surcharger dans les sous-classes
    def specific_reset(self):
        pass

    def custom_reset(self):
        self.lidar_sensor = self.build_lidar_sensor()
        self.pid_controller = DSLPIDControl(drone_model=self.DRONE_MODEL)
        self.time_elapsed_text_id = None
        self.specific_reset()

    def reset(self, seed : int = None, options : dict = None):
        self.custom_reset()
        return super().reset(seed, options)
    
    # Méthode appelée par la classe mère (BaseAviary) au moment de reset() l'environnement
    # Ajoute les obstacles dans l'environnement
    # Surcharger dans les sous-classes pour ajouter des obstacles
    def _addObstacles(self):
        pass

    # Méthode utilitaire permettant d'ajouter un obstacle fixe simplement à l'environnement
    # Les paramètres sont la position du centre de l'obstacle, sa taille totale, sa couleur et sa rotation
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

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos,
            physicsClientId=self.CLIENT,
            baseOrientation=p.getQuaternionFromEuler(rotation_rpy)
        )

    # retourne la position réelle du drone dans la simulation (ground-truth)
    def get_real_drone_pos(self): return self.pos[0,:] 
    
    # retourne la position estimée du drone dans la simulation
    # pour ajouter du bruit ou une manière alternative d'obtenir cette information, c'est cette méthode qu'il faut modifier/surcharger
    def get_estimated_drone_pos(self): return self.get_real_drone_pos()
    
    # retourne la vitesse réelle (vecteur 3D) du drone dans la simulation (ground-truth)
    def get_real_drone_velocity(self): return self.vel[0,:]
    
    # retourne la vitesse (vecteur 3D) estimée du drone dans la simulation
    # pour ajouter du bruit ou une manière alternative d'obtenir cette information, c'est cette méthode qu'il faut modifier/surcharger
    def get_estimated_drone_velocity(self): return self.get_real_drone_velocity()
    
    # retourne l'orientation réelle du drone en RPY (roll, pitch, yaw)
    def get_real_drone_attitude(self): return self.rpy[0,:]
    
    # retourne une estimation de l'attitude du drone en RPY (roll, pitch, yaw)
    # pour ajouter du bruit ou une manière alternative d'obtenir cette information, c'est cette méthode qu'il faut modifier/surcharger
    def get_estimated_drone_attitude_rpy(self): return self.get_real_drone_attitude()
    
    # retourne une estimation de l'attitude du drone au format quaternion
    # pour ajouter du bruit ou une manière alternative d'obtenir cette information, c'est cette méthode qu'il faut modifier/surcharger
    def get_estimated_drone_attitude_quaternion(self): return p.getQuaternionFromEuler(self.get_estimated_drone_attitude_rpy())
    
    # retourne la vitesse angulaire réelle en RPY du drone
    def get_real_angular_vel_rpy(self): return self.ang_v[0,:]
    
    # retourne la vitesse angulaire estimée en RPY du drone
    # pour ajouter du bruit ou une manière alternative d'obtenir cette information, c'est cette méthode qu'il faut modifier/surcharger
    def get_estimated_angular_vel_rpy(self): return self.get_real_angular_vel_rpy()
    
    # retourne un vecteur représentant une estimation de l'état du drone
    # dans l'état actuel du projet, l'état estimé est le même que l'état réel mais cette structure permet une amélioration future
    def get_estimated_drone_state(self):
        state = np.array(self._getDroneStateVector(0))
        state[0:3] = self.get_estimated_drone_pos()
        state[3:7] = self.get_estimated_drone_attitude_quaternion()
        state[7:10] = self.get_estimated_drone_attitude_rpy()
        state[10:13] = self.get_estimated_drone_velocity()
        state[13:16] = self.get_estimated_angular_vel_rpy()
        state[16:20] = np.zeros((4,)) # correspond à la dernière action (RPM des hélices), mise à zéro ici 
        return state

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
    
    # Applique la physique au drone
    def apply_physics(self, action_rpms):
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # stockage des informations sur la physique de l'environnement
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()

            # mise à jour de la physique en fonction du mode de physique défini (self.PHYSICS)
            DRONE_ID = 0
            if self.PHYSICS == Physics.PYB:
                self._physics(action_rpms[DRONE_ID, :], DRONE_ID)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(action_rpms[DRONE_ID, :], DRONE_ID)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(action_rpms[DRONE_ID, :], DRONE_ID)
                self._groundEffect(action_rpms[DRONE_ID, :], DRONE_ID)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(action_rpms[DRONE_ID, :], DRONE_ID)
                self._drag(self.last_clipped_action[DRONE_ID, :], DRONE_ID)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(action_rpms[DRONE_ID, :], DRONE_ID)
                self._downwash(DRONE_ID)
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(action_rpms[DRONE_ID, :], DRONE_ID)
                self._groundEffect(action_rpms[DRONE_ID, :], DRONE_ID)
                self._drag(self.last_clipped_action[DRONE_ID, :], DRONE_ID)
                self._downwash(DRONE_ID)
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = action_rpms
        # stockage des informations sur la physique de l'environnement
        self._updateAndStoreKinematicInformation()

    def step_pid_only(self, action):
        # action convertie en 4 valeurs de RPM (une pour chaque hélice du drone)
        # NUM_DRONES vaut toujours 1 dans cet environnement
        action_RPMs = np.reshape(self._preprocessAction(action), (DRONES_COUNT, 4))
        self.apply_physics(action_RPMs)
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)

    def step_observation_only(self):
        # préparation des valeurs de retour
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, reward, terminated, truncated, info
    
    # Surcharge de step() de BaseAviary
    # Légèrement adapté pour correspondre au fonctionnement de l'environnement
    def step(self, action):
        if self.GUI:
            if self.time_elapsed_text_id is not None:
                p.removeUserDebugItem(self.time_elapsed_text_id)
            self.time_elapsed_text_id = p.addUserDebugText(str(round(self.get_elapsed_time(),1)) + ' / ' + str(round(self.max_episode_duration,1)) + ' seconds.', self.get_real_drone_pos(), textColorRGB=[0, 1, 0], textSize=1.5)

        assert(self.CTRL_FREQ % self.action_freq == 0)
        
        pid_steps_per_control = int(self.CTRL_FREQ / self.action_freq)

        for _ in range(pid_steps_per_control):
            self.step_pid_only(action)
        
        return self.step_observation_only()

    # Retourne une observation de l'environnement (état du drone et de ses capteurs)
    def _computeObs(self):
        self.lidar_sensor.update(1. / self.CTRL_FREQ)

        observation = np.zeros(self._observationSpace().shape)
        observation[0:self.lidar_rays_count] = self.lidar_sensor.read_normalized_distances()

        return observation
    
    def reset_map(self):
        if not self.enable_mapping:
            raise('error : could not create map; mapping is disabled.')
        
        size = np.array([MAP_WIDTH_METERS,MAP_WIDTH_METERS,MAP_HEIGHT_METERS])
        origin = size / 2
        self.map = Map(xyz_size=size, origin_offset=origin, resolution_voxels_per_unit=MAP_RESOLUTION)
        
    def update_map(self):
        if not self.enable_mapping:
            raise('error : could not update map; mapping is disabled.')
        
        self.map.add_scan(local_scan_points=self.lidar_sensor.read_local_points(), sensor_position=self.get_estimated_drone_pos(), max_distance=self.lidar_max_distance)            
    
    def rotate_vector_by_rpy(self, pos, rpy): # todo move to other file
        orientation_quat = p.getQuaternionFromEuler(rpy)
        rot_matrix = R.from_quat(orientation_quat).as_matrix()
        return rot_matrix @ np.array(pos)

    # Méthode appellée automatiquement par la classe parente.
    # Convertit l'action (voir _actionSpace()) en 4 valeurs : RPM des quatres moteurs.
    # Dans notre cas, l'action (vecteur vitesse cible) est convertie en RPM grâce à l'implémentation
    # de DSLPIDControl, un contrôleur PID implémenté par gym-pybullet-drones et qui provient de UTIAS DSL (Dynamic Systems Lab)
    def _preprocessAction(self, action):
        # la position cible est la position actuelle du drone
        # cela a pour effet de désactiver la position cible, on contrôle le déplacement avec un vecteur de vitesse (déclaré plus bas)
        target_pos = self.get_estimated_drone_pos()

        # pareil pour l'attitude cible, on contrôlera la rotation avec un vecteur de vitesse de rotation    
        target_rpy = self.get_estimated_drone_attitude_rpy()

        # la direction cible dépend de l'action choisie (voir classe Action)
        # si l'action n'implique pas de déplacement, la direction retournée est un vecteur nul
        # si le drone se déplace vers le haut ou le bas, sa vitesse cible est diminuée
        target_dir = np.array(action_to_direction(action))
        speed = self.max_drone_velocity if action == Action.FORWARD else self.max_drone_velocity * self.drone_up_and_down_speed_multiplier
        target_vel = target_dir * speed

        # ajustement du vecteur vitesse cible en fonction de l'attitude du drone
        target_vel = self.rotate_vector_by_rpy(target_vel, target_rpy)

        # calcul de la vitesse de rotation cible
        rotation_dir = np.array(action_to_rotation_vector(action))
        target_rpy_rates = rotation_dir * DEFAULT_DRONE_MAX_ROTATION_RATE

        # calcul des nouveaux RPMs des hélices en fonction de l'état actuel estimé du drone et des valeurs cibles (vitesse de déplacement et de rotation)
        state = self.get_estimated_drone_state()
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
    
    # retourne True si le drone est en contact avec un obstacle, sinon False
    def check_for_collisions(self):
        if p.getContactPoints(bodyA=self.DRONE_IDS[0]):
            return True
        return False

    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        raise Exception("not implemented, must be implemented in subclasses")
    
    # Retourne True si l'épisode doit être considéré comme étant terminé.
    def _computeTerminated(self):
        return self.check_for_collisions()    

    def _computeTruncated(self):
        return self.get_elapsed_time() >= self.max_episode_duration
    
    def _computeInfo(self):
        return {'info':None}
