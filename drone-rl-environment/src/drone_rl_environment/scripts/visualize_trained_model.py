# Ce script permet de visualiser un modèle entraîné dans un environnement visuel
# Permet de débugger ou de simplement voir la progression d'un modèle.
# Utilisation : poetry run python .\src\drone_rl_environement\visualize_model.py <NOM DU MODÈLE>.zip
# Note : le modèle doit se trouver dans le dossier .\models\

import argparse
from drone_rl_environment.rl_utils import *
from pathlib import Path

def visualize_model(model_filename):
    print('Creating env...')
    eval_env = VecMonitor(DummyVecEnv([make_gui_env]))
    print('Env created.')

    print('Loading model from file...')
    model = load_model(model_filename, env=eval_env)
    print('...Done !')

    ACTION_FREQ = eval_env.envs[0].CTRL_FREQ / DEFAULT_ACTION_FREQ

    for i in range(999):
        obs = eval_env.reset()
        eval_env.envs[0].reset_map()
        start_time = time.time()
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            if step == 0 or step % ACTION_FREQ == 0:
                action, _ = model.predict(obs, deterministic=True)
                #obs, rewards, dones, infos = eval_env.step(action)
            
            # todo : refactor cette boucle
            eval_env.envs[0].step_pid_only(action)
            obs[0], reward, terminated, truncated, info = eval_env.envs[0].step_observation_only()
            done = terminated or truncated
            total_reward += reward

            if not done:
                eval_env.envs[0].update_map()

            ctrl_ts = eval_env.get_attr("CTRL_TIMESTEP")[0]
            sync(step, start_time, ctrl_ts)
            step += 1

        print('Épisode terminé. Reward total :', round(total_reward,1))
        eval_env.envs[0].save_map(suffix='-'+str(i)) # todo : sauvegarde dans un dossier fixe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise le modèle donné en paramètre dans un environnement 3D")
    parser.add_argument("model_filename", help="Fichier .zip du modèle pré-entrainé. Le fichier doit se trouver dans le répertoire .\\models\\")
    args = parser.parse_args()
    visualize_model(args.model_filename)