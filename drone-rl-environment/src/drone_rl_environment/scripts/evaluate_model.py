import argparse
from drone_rl_environment.rl_utils import *
from pathlib import Path

def make_eval_env():
    return FlyAwayTunnelEnv(gui=False,
                            enable_random_tunnel_rotation=True,
                            enable_mapping=False,
                            initial_xyz_position=INIT_XYZS,
                            initial_rpy_attitude=INIT_RPYS,
                            tunnel_width=1,
                            tunnel_height=1,
                            lidar_rays_count=10,
                            max_episode_duration=20,
                            tunnel_length=30,
                            enable_lidar_rays_debug=False)

def visualize_model(model_filename):
    path = Path.cwd() / MODELS_FOLDER / model_filename # todo : faire ça proprement
    eval_env = VecMonitor(DummyVecEnv([make_eval_env]))

    print('Loading model from file...')
    model = PPO.load(path, env=eval_env) 
    print('...Done !')

    ACTION_FREQ = eval_env.envs[0].CTRL_FREQ / LEARNING_FREQ

    episodes_rewards = []
    episodes_durations = []
    EVAL_EPISODES_COUNT = 10
    for i in range(EVAL_EPISODES_COUNT):
        obs = eval_env.reset()
        start_time = time.time()
        done = False
        step = 0

        dones = [False]

        total_ep_reward = 0

        ctrl_ts = eval_env.get_attr("CTRL_TIMESTEP")[0]
        
        while not done:
            if step == 0 or step % ACTION_FREQ == 0:
                action, _ = model.predict(obs, deterministic=True)
                #obs, rewards, dones, infos = eval_env.step(action)
                o, r, ter, trunc, inf = eval_env.envs[0].step_observation_only()
                obs[0] = o
                dones[0] = ter or trunc
                done = dones[0]
                total_ep_reward += r
            
            # todo : refactor cette boucle
            eval_env.envs[0].step_pid_only(action)
            
            step += 1

        episodes_rewards.append(total_ep_reward)
        episodes_durations.append(ctrl_ts*(step - 1))

        print(f'Épisode N° {i+1}/{EVAL_EPISODES_COUNT} terminé. Durée : {ctrl_ts*(step - 1)} secs. Reward : {total_ep_reward}.')
        print('épisode terminé', step)
        
    eval_env.close()

    reward_avg = sum(episodes_rewards) / len(episodes_rewards)
    duration_avg = sum(episodes_durations) / len(episodes_durations)

    print(sum(episodes_rewards))
    print(len(episodes_rewards))
    
    print('Résumé :')
    print(f'- Reward moyen : {reward_avg}')
    print(f'- Durée moyenne de l\'épisode : {duration_avg}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise le modèle donné en paramètre dans un environnement 3D")
    parser.add_argument("model_filename", help="Fichier .zip du modèle pré-entrainé. Le fichier doit se trouver dans le répertoire .\\models\\")
    args = parser.parse_args()
    visualize_model(args.model_filename)