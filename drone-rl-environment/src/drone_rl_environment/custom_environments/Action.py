from enum import IntEnum, auto

class Action(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return start + count - 1

    STOP = auto()
    FORWARD = auto()
    #DRIFT_LEFT = 2
    #DRIFT_RIGHT = 3
    UP = auto()
    DOWN = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()
    ACTIONS_COUNT = auto()

def action_to_direction(actionIndex):
    match actionIndex:
        case Action.STOP: return [0,0,0] # stop
        case Action.FORWARD: return [1,0,0] # avant
        #case Action.DRIFT_LEFT: return [0,1,0]
        #case Action.DRIFT_RIGHT: return [0,-1,0]
        case Action.UP: return [0,0,1] # haut
        case Action.DOWN: return [0,0,-1] # bas
        case Action.ROTATE_LEFT: return [0,0,0] # rotation gauche
        case Action.ROTATE_RIGHT: return [0,0,0] # rotation droite
        case _: raise('unknown enum value')