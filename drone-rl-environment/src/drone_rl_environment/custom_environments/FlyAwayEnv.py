import math
from .BaseRLSingleDroneEnv import *

class FlyAwayEnv(BaseRLSingleDroneEnv):
    def specific_reset(self):
        self.last_distance = 0

    def _addObstacles(self):
        pass # no obstacles

    def compute_distance_reward(self, delta_distance):
        MAX_DELTA = self.max_drone_velocity / self.action_freq
        MAX_DISTANCE_REWARD_PER_SECOND = 200
        MAX_NEGATIVE_REWARD_PER_SECOND = 500

        EXP_FACTOR = 4

        # normalisation entre -1 et 1 de delta_distance
        x = max(-1.0, min(1.0, delta_distance / MAX_DELTA))

        if x >= 0.0: # reward exponentiel pour les valeurs positives
            reward = MAX_DISTANCE_REWARD_PER_SECOND * (math.exp(EXP_FACTOR * x) - 1.0) / (math.e**EXP_FACTOR - 1.0)
        else: # reward quadratique négatif pour les retours en arrière
            reward = -MAX_NEGATIVE_REWARD_PER_SECOND * (abs(x) ** 2)

        # on divise le reward par la fréquence de l'environnement pour normaliser
        return reward / self.action_freq
    
    def distance_from_starting_point(self):
        return np.linalg.norm(self.get_estimated_drone_pos() - self.INIT_XYZS[0,:])
    
    def compute_delta_distance(self):
        distance = self.distance_from_starting_point()
        delta_distance = distance - self.last_distance
        self.last_distance = distance
        return delta_distance
    
    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        reward = self.compute_distance_reward(self.compute_delta_distance())

        DANGER_RADIUS = 0.2 # 20 cm danger radius #todo : constante ailleurs
        min_dist = self.lidar_sensor.read_distances().min()
        if min_dist < DANGER_RADIUS:
            reward -= 500 * (1 - min_dist / DANGER_RADIUS) / self.CTRL_FREQ

        # en cas de collision, reward extrêmement négatif
        if self.check_for_collisions():
            reward -= 5000
        
        return reward 