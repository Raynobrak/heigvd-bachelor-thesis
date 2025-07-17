# Ce script permet de visualiser un modèle entraîné dans un environnement visuel
# Permet de débugger ou de simplement voir la progression d'un modèle.
# Utilisation : poetry run python .\src\drone_rl_environement\visualize_model.py <NOM DU MODÈLE>.zip
# Note : le modèle doit se trouver dans le dossier .\models\

import argparse
from rl_utils import *
from pathlib import Path

def make_gui_env():
    return FlyAwayTunnelEnv(gui=False,
                            enable_random_tunnel_rotation=True,
                            enable_mapping=True,
                            initial_xyz_position=INIT_XYZS,
                            initial_rpy_attitude=INIT_RPYS,
                            tunnel_width=1,
                            tunnel_height=1,
                            lidar_rays_count=10,
                            max_episode_duration=50,
                            enable_lidar_rays_debug=False)

def visualize_model(model_filename):
    path = Path.cwd() / 'models' / model_filename # todo : faire ça proprement

    eval_env = VecMonitor(DummyVecEnv([make_gui_env]))          # n_envs = 1

    #todo make this work 
    print('Loading model from file...')
    model = DQN.load(path, env=eval_env)  # OK même si le modèle fut entraîné à 8 envs
    print('...Done !')

    for i in range(30):
        obs = eval_env.reset()                                   # shape = (1, obs_dim)
        eval_env.envs[0].reset_map()
        start_time = time.time()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            done = dones[0]

            if not done:
                eval_env.envs[0].update_map()

            ctrl_ts = eval_env.get_attr("CTRL_TIMESTEP")[0]
            #sync(step, start_time, ctrl_ts)
            step += 1

        print('épisode terminé')
        eval_env.envs[0].save_map(filename='map-'+str(i)+'.png')
        #eval_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise le modèle donné en paramètre dans un environnement 3D")
    parser.add_argument("model_filename", help="Fichier .zip du modèle pré-entrainé. Le fichier doit se trouver dans le répertoire .\\models\\")
    args = parser.parse_args()
    visualize_model(args.model_filename)