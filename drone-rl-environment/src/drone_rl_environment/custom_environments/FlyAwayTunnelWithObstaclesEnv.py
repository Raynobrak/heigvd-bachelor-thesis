from .FlyAwayTunnelEnv import *

class FlyAwayTunnelWithObstaclesEnv(FlyAwayTunnelEnv):
    def _addObstacles(self):
        super()._addObstacles() # ajout du tunnel

        raise Exception('not implemented')
        # todo : génération procédurale d'obstacles ou de virages dans le tunnel
        
