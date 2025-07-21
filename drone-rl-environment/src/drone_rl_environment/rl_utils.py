#
# Ce fichier contient des fonctions encapsulant certaines étapes répétitives de l'apprentissage par renforcement
# en particulier, l'entraînement, la visualisation et la sauvegarde du modèle.
# Le but de ces fonctions est de pouvoir les utiliser pour facilement "scripter" un scénario en écrivant un minimum de code
#

import time
import os
from pathlib import Path
from stable_baselines3 import PPO, DDPG, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import get_linear_fn

from drone_rl_environment.custom_environments.FlyAwayCeilingEnv import *
from drone_rl_environment.custom_environments.FlyAwayTunnelEnv import *
from drone_rl_environment.custom_environments.Action import *
from gym_pybullet_drones.utils.utils import sync

# position de départ et attitude (roll, pitch, yaw)
INIT_XYZS = np.array([[0,0,0.2]])
INIT_RPYS = np.array([np.zeros((3,))])

SLSH = '\\'
MODELS_FOLDER = 'models'
NUM_SUBPROC_ENVS = 6
TIMESTEPS_PER_EPOCH = 4096
STATS_WINDOW_SIZE = 20
TENSORBOARD_LOGS_FOLDER = "./tensorboard-logs/"

def make_gui_env():
    return FlyAwayTunnelEnv(gui=True,
                            enable_random_tunnel_rotation=True,
                            enable_mapping=True,
                            initial_xyz_position=INIT_XYZS,
                            initial_rpy_attitude=INIT_RPYS,
                            tunnel_length=30,
                            tunnel_width=1,
                            tunnel_height=1,
                            lidar_rays_count=10,
                            max_episode_duration=30,
                            enable_lidar_rays_debug=True)

def make_env(evaluation=False):
    def _init():
        return FlyAwayTunnelEnv(
            enable_random_tunnel_rotation=True,
            initial_xyz_position=INIT_XYZS,
            initial_rpy_attitude=INIT_RPYS,
            max_episode_duration=30,
            gui=evaluation,
            tunnel_length=30,
            tunnel_width=1,
            tunnel_height=1,
            lidar_rays_count=10,
            enable_lidar_rays_debug=False,
            enable_mapping=evaluation
        )
    return _init

def create_environment(evaluation=False):
    if evaluation:
        return DummyVecEnv([make_env(evaluation=False)])
    else:
        num_envs = NUM_SUBPROC_ENVS
        vec_env = SubprocVecEnv([make_env(evaluation=False) for _ in range(num_envs)])
        vec_env = VecMonitor(vec_env)
        return vec_env

def get_dummy_env():
    return create_environment(evaluation=False)

# créé un nouveau modèle de RL
def create_model():
    dummy_env = get_dummy_env() # si aucun environnement n'est spécifié, créé un environnement artificiel qui est immédiatement détruit. c'est juste pour que ça compile
    model = PPO(
        'MlpPolicy',
        env=dummy_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=1e-4,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=TENSORBOARD_LOGS_FOLDER,
        stats_window_size=STATS_WINDOW_SIZE
    )
    """model = DDPG(
        'MlpPolicy',
        env=dummy_env,
        verbose=1,
        batch_size=256,
        tensorboard_log=TENSORBOARD_LOGS_FOLDER,
    )"""
    #model = DQN("MlpPolicy", dummy_env, verbose=1)
    
    """model = DQN(
        "MlpPolicy",
        dummy_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=1000,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[64,64]),
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGS_FOLDER
    )"""
    """model = DQN(
        "MlpPolicy",
        dummy_env,
        learning_rate=2.5e-4,
        buffer_size=500_000,
        batch_size=128,
        learning_starts=5_000,
        train_freq=(4, "step"),
        gradient_steps=4,
        target_update_interval=10_000,
        gamma=0.995,
        exploration_initial_eps=1.0,
        exploration_fraction=0.10,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        max_grad_norm=10,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGS_FOLDER,
    )"""
    """model = A2C(
    policy="MlpPolicy",
    env=dummy_env,
    learning_rate=1e-3,
    gamma=0.99,
    max_grad_norm=10,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
    tensorboard_log=TENSORBOARD_LOGS_FOLDER
    )"""

    return model

# charge un modèle à partir d'un fichier
def load_model(filename, env=None):
    path = Path.cwd() / MODELS_FOLDER / filename
    return PPO.load(path, env=create_environment(evaluation=True) if env is not None else env)

# sauvegarde un modèle dans le dossier donné
# par défaut, le dossier est le dossier /models/
def save_model(model, prefix, folder = MODELS_FOLDER):
    fname = f"{prefix}_{time.strftime('%d-%H-%M-%S')}"
    path = Path.cwd() / folder / fname
    model.save(path)

# entraîne le modèle donné dans un environnement d'entraînement pendant n steps
def train_model_in_environment(model, timesteps=TIMESTEPS_PER_EPOCH, training_env=None, tb_run_name=None):
    if training_env is None:
        training_env = create_environment(evaluation=False)

    print(f"Training model for {timesteps} steps...")
    model.set_env(training_env)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=None, reset_num_timesteps=False, tb_log_name=tb_run_name)
    training_env.close() # nécessaire de fermer l'environnement car pybullet ne peut pas avoir plusieurs environnements de simulation en parallèle (entraînement + évaluation/visualisation)

def visualize_model_in_environment(model, num_episodes=5):
    for i in range(num_episodes):
        eval_env = create_environment(evaluation=True)
        model.set_env(eval_env)

        action = np.zeros(eval_env.action_space.shape)
        START = time.time()
        terminated, truncated = False, False
        step = 0

        DEMO_MODE = 1 # todo : temporaire, voir pour enlever ou faire une fonction à part pour une démo

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            if DEMO_MODE:
                if step < eval_env.CTRL_FREQ * 1:
                    action = Action.STOP
                elif step < eval_env.CTRL_FREQ * 2:
                    action = Action.FORWARD
                elif step < eval_env.CTRL_FREQ * 3:
                    action = Action.ROTATE_LEFT
                elif step < eval_env.CTRL_FREQ * 4:
                    action = Action.ROTATE_RIGHT
                elif step < eval_env.CTRL_FREQ * 5:
                    action = Action.DRIFT_LEFT
                elif step < eval_env.CTRL_FREQ * 6:
                    action = Action.DRIFT_RIGHT
                elif step < eval_env.CTRL_FREQ * 7:
                    action = Action.UP
                elif step < eval_env.CTRL_FREQ * 8:
                    action = Action.DOWN
                else:
                    action = Action.STOP
            else:
                action, _ = model.predict(obs) # prédiction du modèle sur l'action à effectuer

            # synchronisation de l'affichage de la simulation
            # seulement si on veut l'interface grahique
            sync(step, START, eval_env.CTRL_TIMESTEP)
            step += 1
        eval_env.close()