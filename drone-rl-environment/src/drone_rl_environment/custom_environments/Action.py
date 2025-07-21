from enum import IntEnum, auto

class Action(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return start + count - 1

    STOP = auto()
    FORWARD = auto()
    UP = auto()
    DOWN = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()
    ACTIONS_COUNT = auto()

# retourne la direction de déplacement impliquée par l'action
# si l'action n'implique pas de déplacement, un vecteur nul est retourné
def action_to_direction(actionIndex):
    match actionIndex:
        case Action.STOP: return [0,0,0] # stop
        case Action.FORWARD: return [1,0,0] # avant
        case Action.UP: return [0,0,1] # haut
        case Action.DOWN: return [0,0,-1] # bas
        case Action.ROTATE_LEFT: return [0,0,0] # rotation gauche
        case Action.ROTATE_RIGHT: return [0,0,0] # rotation droite
        case _: raise('unknown enum value')

# retourne la direction de rotation (RPY) impliquée par l'action
# si l'action n'implique pas de rotation, un vecteur nul est retourné
def action_to_rotation_vector(actionIndex):
    match actionIndex:
        case Action.STOP: return [0,0,0] # stop
        case Action.FORWARD: return [0,0,0] # avant
        case Action.UP: return [0,0,0] # haut
        case Action.DOWN: return [0,0,0] # bas
        case Action.ROTATE_LEFT: return [0,0,1] # rotation gauche
        case Action.ROTATE_RIGHT: return [0,0,-1] # rotation droite
        case _: raise('unknown enum value')