from src.sensors.lidar import *
from src.constants import *

import numpy as np

from breezyslam.algorithms import RMHC_SLAM, Deterministic_SLAM
from breezyslam.sensors import *

import roboviz as rv

# todo : définir distance minimale de détection
class Lidar(Laser):
    def __init__(self):
        Laser.__init__(self, NB_LIDAR_ANGLES, SIMULATION_FPS, 360, 0, 0, 0)

class Map:
    def __init__(self):
        self.lidar = Lidar()
        self.mapbytes = bytearray(2*WINDOW_WIDTH*2*WINDOW_WIDTH)

        self.slam = RMHC_SLAM(self.lidar, 2*WINDOW_WIDTH, 2*px_to_meters(WINDOW_WIDTH), map_quality=5, hole_width_mm=400)

        self.starting_pos = vec(self.slam.position.x_mm, self.slam.position.y_mm)
        self.viz = rv.MapVisualizer(2*WINDOW_WIDTH, 2*px_to_meters(WINDOW_WIDTH), 'SLAM')

    def update_slam(self, lidar_scan_millimeters, motion_estimation_millimeters):
        #todo fix : estimated motion doit être un tuple (dxy, theta, dt seconds)
        self.slam.update(scans_mm=lidar_scan_millimeters, pose_change=None)

    def get_estimated_total_movement_mm(self):
        x, y, theta = self.slam.getpos()
        return vec(x,y) - self.starting_pos
    
    def get_estimated_position_in_window_px(self, window_size_px, starting_position_px):
        pos = mm_to_px(self.get_estimated_total_movement_mm())
        pos.y = -pos.y
        pos += starting_position_px
        return pos
    
    def window_to_map(self, win_coords):
        pos = vec(win_coords.x, -win_coords.y) + mm_to_px(self.starting_pos)
        self.test = px_to_meters(pos)


    def get_map(self):
        self.slam.getmap(self.mapbytes)
        return self.mapbytes
    
    def display(self):
        m = self.get_map()
        x,y,theta = self.slam.getpos()
        if not self.viz.display(self.test.x, self.test.y, theta, m):
            exit(0)
