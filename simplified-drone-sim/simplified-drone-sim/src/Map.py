from src.sensors.lidar import *
from src.constants import *

class Map:
    def __init__(self):
        self.map = []

    def add_scan_at_pos(self, exact_pos, lidar_data):
        points = lidar_data_to_points(lidar_data, exact_pos, ignore_max_distances=True)
        for p in points:
            self.map.append(p)

    def get_map_points(self):
        return self.map
