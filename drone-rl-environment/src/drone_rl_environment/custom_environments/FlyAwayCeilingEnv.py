from .FlyAwayEnv import *

class FlyAwayCeilingEnv(FlyAwayEnv):
    def _addObstacles(self):
        super()._addObstacles()
        self.add_fixed_obstacle(center_pos=[0,0,1], size=[50,50,0.2], rgba_color=[0,0,1,0.4])
