from src.sensors.lidar import *
from src.constants import *

import numpy as np

from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import *

import roboviz as rv

# todo : définir distance minimale de détection
#def __init__(self, scan_size, scan_rate_hz, detection_angle_degrees, distance_no_detection_mm, detection_margin=0, offset_mm=0):
class Lidar(Laser):
    def __init__(self):
        Laser.__init__(self, NB_LIDAR_ANGLES, SIMULATION_FPS, 360, 0, 0, 0)

class Map:
    def __init__(self):
        self.map = []

        self.lidar = Lidar()
        self.mapbytes = bytearray(WINDOW_WIDTH*WINDOW_WIDTH)

        # def __init__(self, laser, map_size_pixels, map_size_meters, 
        # map_quality=_DEFAULT_MAP_QUALITY, hole_width_mm=_DEFAULT_HOLE_WIDTH_MM)
        self.slam = RMHC_SLAM(self.lidar, WINDOW_WIDTH, px_to_meters(WINDOW_WIDTH), map_quality=100) 

        self.viz = rv.MapVisualizer(WINDOW_WIDTH, px_to_meters(WINDOW_WIDTH), 'SLAM')

    def update_slam(self, estimated_motion, lidar_distances):
        #todo fix : estimated motion doit être un tuple (dxy, theta, dt seconds)
        self.slam.update(lidar_distances, estimated_motion)

        x, y, theta = self.slam.getpos()

        return vec(x,y)

    def get_map(self):
        self.slam.getmap(self.mapbytes)
        return self.mapbytes

    def add_scan_at_pos(self, exact_pos, lidar_data):
        points = lidar_data_to_points(lidar_data, exact_pos, ignore_max_distances=True)
        for p in points:
            self.map.append(p)

    def get_map_points(self):
        return self.map
    
    def display(self):
        m = self.get_map()
        if not self.viz.display(0, 0, 0, m):
            exit(0)
