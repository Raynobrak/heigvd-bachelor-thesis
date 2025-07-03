import numpy as np
from src.constants import *

# todo : encapsuler le lidar dans une classe et fixer une fréquence de fonctionnement
# todo : ajouter du bruit

def reverse_multiplier(reverse):
    return -1 if reverse else 1

# sensor_global_pos : global 2D pos of the sensor origin in the world
# obstacles_global : global 2D position and size of obstacles (rectangles) -> list of tuples (x,y,w,h)
# retourne : une liste des distances aux objets les plus proches pour chaque angle du capteur
def emulate_lidar(sensor_global_pos, obstacles, reverse_x = True):
    # définir la fréquence de mesure (nb tour par seconde)
    # définir le nombre d'angles (dans combien de directions différentes envoyer des rayon)
    # définir le "step" de chaque rayon (plus c'est petit, plus c'est précis)
    # définir la distance max

    scan = list()

    for i in range(NB_LIDAR_ANGLES):
        current_angle_rad = np.radians(i * 360 / NB_LIDAR_ANGLES)
        dir = vec(reverse_multiplier(reverse_x) * np.cos(current_angle_rad), np.sin(current_angle_rad))

        obstacle_detected = False

        max_point = sensor_global_pos + dir * MAX_LIDAR_DISTANCE
        max_point = (max_point.x, max_point.y)
        closest_squared = 2*MAX_LIDAR_DISTANCE*MAX_LIDAR_DISTANCE
        for obst in obstacles:
            clipped_line = obst.rect.clipline((sensor_global_pos.x, sensor_global_pos.y), max_point)
            if clipped_line:
                obstacle_detected = True
                (x,y), end = clipped_line
                squared_distance = (vec(x,y) - sensor_global_pos).magnitude_squared()
                if squared_distance < closest_squared:
                    closest_squared = squared_distance
        if obstacle_detected:
            scan.append(np.sqrt(closest_squared))
        else:
            scan.append(MAX_LIDAR_DISTANCE)
    return scan

def lidar_data_to_points(data, sensor_pos_global, ignore_max_distances=False, reverse_x = True):
    points = []
    for i in range(NB_LIDAR_ANGLES):
        if data[i] != MAX_LIDAR_DISTANCE:
            current_angle_rad = np.radians(i * 360 / NB_LIDAR_ANGLES)

            dir = vec(reverse_multiplier(reverse_x) * np.cos(current_angle_rad), np.sin(current_angle_rad))
            points.append(sensor_pos_global + dir * data[i])

    return points