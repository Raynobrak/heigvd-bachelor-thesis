import numpy as np
from src.constants import *

# todo : encapsuler le lidar dans une classe et fixer une fréquence de fonctionnement

# sensor_global_pos : global 2D pos of the sensor origin in the world
# obstacles_global : global 2D position and size of obstacles (rectangles) -> list of tuples (x,y,w,h)
# retourne : une liste des distances aux objets les plus proches pour chaque angle du capteur
def emulate_lidar(sensor_global_pos, obstacles):
    # définir la fréquence de mesure (nb tour par seconde)
    # définir le nombre d'angles (dans combien de directions différentes envoyer des rayon)
    # définir le "step" de chaque rayon (plus c'est petit, plus c'est précis)
    # définir la distance max

    scan = list()

    LIDAR_STEP = 2

    for i in range(NB_LIDAR_ANGLES):
        current_angle_rad = np.radians(i * 360 / NB_LIDAR_ANGLES)

        dir = vec(np.cos(current_angle_rad), np.sin(current_angle_rad))

        distance = 0
        obstacle_detected = False
        while distance < MAX_LIDAR_DISTANCE and not obstacle_detected:
            point = sensor_global_pos + dir * distance

            for obst in obstacles:
                if obst.contains_point(point):
                    obstacle_detected = True
                    break

            if not obstacle_detected:
                distance += LIDAR_STEP
        
        if obstacle_detected:
            scan.append(distance)
        else:
            scan.append(MAX_LIDAR_DISTANCE)
    return scan

def lidar_data_to_points(data, sensor_pos_global, ignore_max_distances=False):
    points = []
    for i in range(NB_LIDAR_ANGLES):
        if data[i] != MAX_LIDAR_DISTANCE:
            current_angle_rad = np.radians(i * 360 / NB_LIDAR_ANGLES)

            dir = vec(np.cos(current_angle_rad), np.sin(current_angle_rad))
            points.append(sensor_pos_global + dir * data[i])

    return points