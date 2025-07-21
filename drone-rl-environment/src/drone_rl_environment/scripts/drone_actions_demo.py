# Petit script permettant de visualiser les contrôles haut-nvieau du drone
# C'est à dire, les actions STOP, FORWARD, UP, DOWN, ROTATE_LEFT, ROTATE_RIGHT définies dans Action.py

from drone_rl_environment.rl_utils import *

u = Action.UP
d = Action.DOWN
f = Action.FORWARD
rl = Action.ROTATE_LEFT
rr = Action.ROTATE_RIGHT
s = Action.STOP

# liste d'actions à effectuer
ACTION_SEQUENCE = [
    u,u,u,u,u,
    s,s,s,s,s,
    d,d,s,s,s,
    f,f,f,f,f,
    rl,rl,rl,rl,rl,
    f,rr,rr,rr,f,
    rr,rr,rr,f,f,
    rl,f,rl,f,rl,
    f,rl,f,rl,f,
    rl,f,rl,f,f,
    s,s,s,s,s,
]

def run_demo():
    eval_env = FlyAwayEnv(
        max_episode_duration=9999,
        gui=True,
        lidar_rays_count=10,
        enable_lidar_rays_debug=False,
        enable_mapping=False,
        action_freq=5
    )

    action = Action.STOP
    START = time.time()
    terminated, truncated = False, False
    step = 0

    PID_STEPS_BETWEEN_ACTIONS = eval_env.CTRL_FREQ / eval_env.action_freq

    while not (terminated or truncated):
        if step == 0 or step % PID_STEPS_BETWEEN_ACTIONS == 0:
            # on boucle à travers les actions
            action_idx = int((step / PID_STEPS_BETWEEN_ACTIONS) % len(ACTION_SEQUENCE))
            action = ACTION_SEQUENCE[action_idx]
            
        eval_env.step_pid_only(action)
        obs, reward, terminated, truncated, info = eval_env.step_observation_only()

        # synchronisation de l'affichage de la simulation
        sync(step, START, eval_env.CTRL_TIMESTEP)
        step += 1
    eval_env.close()

if __name__ == '__main__':
    run_demo()