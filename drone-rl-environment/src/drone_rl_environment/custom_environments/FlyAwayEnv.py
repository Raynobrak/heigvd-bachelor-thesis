from .BaseRLSingleDroneEnv import *

class FlyAwayEnv(BaseRLSingleDroneEnv):
    # Initialise l'environnement
    def __init__(self,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 gui=False,
                 ):
        super().__init__(initial_xyz_position, initial_rpy_attitude, max_drone_velocity, max_episode_duration, gui)

    def _addObstacles(self):
        pass # no obstacles
    
    # Calcule le reward en fonction de l'état actuel de l'environnement
    def _computeReward(self):
        distance_from_starting_point = np.linalg.norm(self.get_estimated_drone_pos() - self.INIT_XYZS[0,:])

        reward = 10 * distance_from_starting_point / self.CTRL_FREQ # reward += distance par rapport au point de départ / seconde. donc si le drone est à 10 mètres, c'est 10 "points" par seconde.

        # en cas de collision, reward extrêmement négatif + fin de l'épisode
        if self.check_for_collisions():
            reward -= 5000
        
        return reward 