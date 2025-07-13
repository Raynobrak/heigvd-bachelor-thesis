from enum import IntEnum

class Action(IntEnum):
    STOP = 0
    FORWARD = 1
    DRIFT_LEFT = 2
    DRIFT_RIGHT = 3
    UP = 4
    DOWN = 5
    ROTATE_LEFT = 6
    ROTATE_RIGHT = 7
    ACTIONS_COUNT = 8

def action_to_direction(actionIndex):
    match actionIndex:
        case Action.STOP: return [0,0,0] # stop
        case Action.FORWARD: return [1,0,0] # avant
        case Action.DRIFT_LEFT: return [0,1,0] # haut
        case Action.DRIFT_RIGHT: return [0,-1,0] # bas
        case Action.UP: return [0,0,1] # gauche
        case Action.DOWN: return [0,0,-1] # droite
        case Action.ROTATE_LEFT: return [0,0,0] # rotation gauche
        case Action.ROTATE_RIGHT: return [0,0,0] # rotation droite
        case _: raise('unknown enum value')