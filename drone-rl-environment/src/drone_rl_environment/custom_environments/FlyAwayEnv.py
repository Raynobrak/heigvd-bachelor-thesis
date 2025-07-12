from .BaseRLSingleDroneEnv import *

class FlyAwayEnv(BaseRLSingleDroneEnv):
    def specific_reset(self):
        self.last_distance = 0

    def _addObstacles(self):
        pass # no obstacles
    
    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        distance_from_starting_point = np.linalg.norm(self.get_estimated_drone_pos() - self.INIT_XYZS[0,:])

        distance_gain = 200 * (distance_from_starting_point - self.last_distance)
        self.last_distance = distance_from_starting_point
        reward = distance_gain

        #print(distance_from_starting_point)

        #reward = 100 * distance_from_starting_point / self.CTRL_FREQ # reward += distance par rapport au point de départ / seconde. donc si le drone est à 10 mètres, c'est 10 "points" par seconde.

        DANGER_RADIUS = 0.2 # 20 cm danger radius #todo : constante ailleurs
        min_dist = self.lidar_sensor.read_distances().min()
        if min_dist < DANGER_RADIUS:
            reward -= 1000 * (1 - min_dist / DANGER_RADIUS) / self.CTRL_FREQ

        # en cas de collision, reward extrêmement négatif + fin de l'épisode
        if self.check_for_collisions():
            reward -= 5000
        
        return reward 