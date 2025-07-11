from .FlyAwayTunnelEnv import *

class FlyAwayTunnelWithObstaclesEnv(FlyAwayTunnelEnv):
    def _addObstacles(self):
        super()._addObstacles() # ajout du tunnel

        # todo : génération procédurale des obstacles dans le tunnel
        
