# Ce script permet de visualiser un modèle entraîné dans un environnement visuel
# Permet de débugger ou de simplement voir la progression d'un modèle.
# Utilisation : poetry run python .\src\drone_rl_environement\visualize_model.py <NOM DU MODÈLE>.zip
# Note : le modèle doit se trouver dans le dossier .\models\

import argparse
from drone_rl_environment.rl_utils import *
from pathlib import Path

def make_gui_env():
    return FlyAwayTunnelEnv(gui=True,
                            enable_random_tunnel_rotation=True,
                            enable_mapping=True,
                            initial_xyz_position=INIT_XYZS,
                            initial_rpy_attitude=INIT_RPYS,
                            tunnel_width=1,
                            tunnel_height=1,
                            lidar_rays_count=10,
                            max_episode_duration=20,
                            enable_lidar_rays_debug=True)
    return FlyAwayEnv(gui=True, lidar_rays_count=10, enable_mapping=True, max_episode_duration=5, enable_lidar_rays_debug=True)

def visualize_model(model_filename):
    path = Path.cwd() / 'models' / model_filename # todo : faire ça proprement

    eval_env = VecMonitor(DummyVecEnv([make_gui_env]))

    #todo make this work 
    print('Loading model from file...')
    model = PPO.load(path, env=eval_env) 
    print('...Done !')

    ACTION_FREQ = eval_env.envs[0].CTRL_FREQ / LEARNING_FREQ

    for i in range(30):
        obs = eval_env.reset()
        eval_env.envs[0].reset_map()
        start_time = time.time()
        done = False
        step = 0

        dones = [False]
        
        while not done:
            if step == 0 or step % ACTION_FREQ == 0:
                action, _ = model.predict(obs, deterministic=True)
                #obs, rewards, dones, infos = eval_env.step(action)
            
            # todo : refactor cette boucle
            eval_env.envs[0].step_pid_only(action)
            o, r, ter, trunc, inf = eval_env.envs[0].step_observation_only()
            obs[0] = o
            dones[0] = ter or trunc

            done = dones[0]

            if not done:
                eval_env.envs[0].update_map()

            ctrl_ts = eval_env.get_attr("CTRL_TIMESTEP")[0]
            sync(step, start_time, ctrl_ts)
            step += 1

        print('épisode terminé', step)
        eval_env.envs[0].save_map(filename='map-'+str(i)+'.png')
        #eval_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise le modèle donné en paramètre dans un environnement 3D")
    parser.add_argument("model_filename", help="Fichier .zip du modèle pré-entrainé. Le fichier doit se trouver dans le répertoire .\\models\\")
    args = parser.parse_args()
    visualize_model(args.model_filename)