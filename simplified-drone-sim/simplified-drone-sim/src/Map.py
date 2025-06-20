from src.sensors.lidar import *
from src.constants import *

import numpy as np
from libraries.toolbuddy_grid_slam.ParticleFilter import *

#todo : renommer les constantes et refactor
NUM_PARTICLES = 10     # nombre de particules

lidar_params = [NB_LIDAR_ANGLES,  0.0, 360.0, MAX_LIDAR_DISTANCE, 1.0,  5.0] # paramètre LIDAR [nb_beams, start_angle_deg, end_angle_deg, max_range, trans_step, rot_step]
initial_bot_pos = np.array([0.0, 0.0, 0.0]) # position intiale : x,y,rotation

# 3) Paramètres de la carte (log-odds occupied / free / max / min)
map_params = [0.4, -0.4, 5.0, -5.0]

class Map:
    def __init__(self):
        self.map = []

        self.gmap = GridMap(map_params)
        self.particle_filter = ParticleFilter(
            initial_bot_pos.copy(),       
            lidar_params,            
            copy.deepcopy(self.gmap),  
            NUM_PARTICLES         
        )

    def update_slam(self, estimated_motion, lidar_distances):
        # todo : estimated_pos doit être calculé et màj l'IMU et prendre en compte la vitesse

        print('debug : begin update')
        # déterminer la commande discrète du mouvement perçu (est-ce qu'on a avancé, reculé ou tourné ? -> control = direction du mouvement)
        control = self._imu_to_control(estimated_motion.x, estimated_motion.y, 0)

        print('debug : feed')

        self.particle_filter.Feed(control, lidar_distances)

        print('debug : resample')
        self.particle_filter.Resampling(lidar_distances)

        # mise à jour de la carte globale à partir du scan et de la particule la plus probable
        print('debug : find best')
        best = np.argmax(self.particle_filter.weights)
        x, y, theta = self.particle_filter.particle_list[best].pos
        # todo : enlever ça ?
        #self.gmap.update(lidar_distances, (x, y, theta))

    def _imu_to_control(self, dx, dy, dtheta):
        ts = lidar_params[4]   # trans_step
        rs = lidar_params[5]   # rot_step

        # Priorité aux translations
        if abs(dx) > abs(dy) and abs(dx) >= ts:
            return 1 if dx > 0 else 2   # 1=avancer, 2=reculer
        # Sinon rotations
        if abs(dtheta) >= rs:
            return 3 if dtheta > 0 else 4   # 3=rot gauche, 4=rot droite
        # Pas de mouvement significatif
        return 0

    def get_drone_position(self):
        best = np.argmax(self.particle_filter.weights)
        x, y, theta = self.particle_filter.particles[best]
        return vec(x,y)

    def add_scan_at_pos(self, exact_pos, lidar_data):
        points = lidar_data_to_points(lidar_data, exact_pos, ignore_max_distances=True)
        for p in points:
            self.map.append(p)

    def get_map_points(self):
        return self.map
