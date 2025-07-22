import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv

from drone_rl_environment.rl_utils import *
from pathlib import Path

CSV_FILE = 'evaluation_data.csv'

def load_evaluation_results(csv_path=CSV_FILE):
    rewards = []
    durations = []
    angles = []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row['reward']))
            durations.append(float(row['duration']))
            angles.append(float(row['angle']))
    return rewards, durations, angles

def plot_evaluation_results(rewards, durations, tunnels_angles, save_csv_path=CSV_FILE):
    assert len(rewards) == len(durations) == len(tunnels_angles)
    
    rewards = np.array(rewards)
    durations = np.array(durations)
    angles = np.array(tunnels_angles)

    # histogramme des rewards
    plt.figure()
    bins_rewards = np.arange(0, 6000 + 100, 100)
    plt.hist(rewards, bins=bins_rewards, alpha=0.7, edgecolor='black')
    plt.axvline(rewards.mean(), color='red', linestyle='--', label=f'Moyenne: {rewards.mean():.2f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--', label=f'Médiane: {np.median(rewards):.2f}')
    plt.xlim(0, 6000)
    plt.title('Histogramme des Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Nombre d\'épisodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # histogramme des durées
    plt.figure()
    bins_durations = np.arange(0, 35 + 2, 2)
    plt.hist(durations, bins=bins_durations, alpha=0.7, edgecolor='black')
    plt.axvline(durations.mean(), color='red', linestyle='--', label=f'Moyenne: {durations.mean():.2f}s')
    plt.axvline(np.median(durations), color='green', linestyle='--', label=f'Médiane: {np.median(durations):.2f}s')
    plt.xlim(0, 35)
    plt.title('Histogramme des Durées des Épisodes')
    plt.xlabel('Durée (s)')
    plt.ylabel('Nombre d\'épisodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # durée vs reward
    plt.figure()
    plt.scatter(durations, rewards, alpha=0.7, edgecolors='k')
    plt.xlim(0, 35)
    plt.ylim(0, 6000)
    plt.title('Durée vs Reward')
    plt.xlabel('Durée de l\'épisode (s)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # angle du tunnel vs reward
    plt.figure()
    plt.scatter(angles, rewards, alpha=0.7, edgecolors='k')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 1000)
    plt.title('Angle du Tunnel vs Reward')
    plt.xlabel('Angle du tunnel (radians)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # sauvegarde des données en csv
    with open(save_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reward', 'duration', 'angle'])
        for r, d, a in zip(rewards, durations, angles):
            writer.writerow([r, d, a])
    print(f'✅ Données sauvegardées dans : {save_csv_path}')

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
    path = Path.cwd() / MODELS_FOLDER / model_filename
    eval_env = VecMonitor(DummyVecEnv([make_eval_env]))

    print('Loading model from file...')
    model = PPO.load(path, env=eval_env) 
    print('...Done !')

    ACTION_FREQ = eval_env.envs[0].CTRL_FREQ / DEFAULT_ACTION_FREQ

    episodes_rewards = []
    episodes_durations = []
    tunnel_angles = []
    EVAL_EPISODES_COUNT = 1000
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
        tunnel_angles.append(eval_env.envs[0].tunnel_angle)

        print(f'Épisode N° {i+1}/{EVAL_EPISODES_COUNT} terminé. Durée : {ctrl_ts*(step - 1)} secs. Reward : {total_ep_reward}.')
        print('épisode terminé', step)
        
    eval_env.close()

    plot_evaluation_results(episodes_rewards, episodes_durations, tunnel_angles)

    reward_avg = sum(episodes_rewards) / len(episodes_rewards)
    duration_avg = sum(episodes_durations) / len(episodes_durations)

    # todo : histogrammes matplotlib
    
    print('Résumé :')
    print(f'- Reward moyen : {reward_avg}')
    print(f'- Durée moyenne de l\'épisode : {duration_avg}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise le modèle donné en paramètre dans un environnement 3D")
    parser.add_argument("model_filename", help="Fichier .zip du modèle pré-entrainé. Le fichier doit se trouver dans le répertoire .\\models\\")
    args = parser.parse_args()
    visualize_model(args.model_filename)