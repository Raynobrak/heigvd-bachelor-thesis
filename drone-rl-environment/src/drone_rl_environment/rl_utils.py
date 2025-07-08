#
# Ce fichier contient des fonctions encapsulant certaines étapes répétitives de l'apprentissage par renforcement
# en particulier, l'entraînement, la visualisation et la sauvegarde du modèle.
# Le but de ces fonctions est de pouvoir les utiliser pour facilement "scripter" un scénario en écrivant un minimum de code
#

import time
import os
from stable_baselines3 import PPO,DDPG

from custom_environments.FlyAwayCeilingEnv import *
from gym_pybullet_drones.utils.utils import sync

# position de départ et attitude (roll, pitch, yaw)
INIT_XYZS = np.array([[0,0,0.2]])
INIT_RPYS = np.array([np.zeros((3,))])

SAVE_FOLDER = ".\\models\\"
TIMESTEPS_PER_EPOCH = 4096

def create_environment(evaluation=False):
    environment = FlyAwayCeilingEnv(
                    initial_xyz_position=INIT_XYZS,
                    initial_rpy_attitude=INIT_RPYS,
                    gui=evaluation,
                    )
    return environment

# créé un nouveau modèle de RL
def create_model(env=None):
    if env is None:
        dummy_env = FlyAwayCeilingEnv() # si aucun environnement n'est spécifié, créé un environnement artificiel qui est immédiatement détruit. c'est juste pour que ça compile
        model = PPO(
            'MlpPolicy',
            env=dummy_env,
            verbose=1,
            n_steps=int(TIMESTEPS_PER_EPOCH / 4),
            batch_size=256,
            n_epochs=20,
            clip_range=0.2,
            ent_coef=0.1,
            tensorboard_log="./ppo_drone_tensorboard/"
        )
        dummy_env.close()
        return model
    return PPO('MlpPolicy', env=env, verbose=1)

# charge un modèle à partir d'un fichier
def load_model(filename, env, use_default_folder=True):
    path = (SAVE_FOLDER + filename) if use_default_folder else filename

    if os.path.isfile(filename):
        return PPO.load(path, env=env)
    else:
        print(f'Error when loading model : "{path}" doesn\'t exist.')
        return None

# charge tous les modèles avec le même préfixe dans le dossier donné
# généralement, les modèles avec le même préfixe correspondent à la même session d'entraînement
def load_models(path, prefix):
    # todo implémenter si besoin
    return None

# sauvegarde un modèle dans path et le nomme de la manière suivante : "prefix_timestamp.zip"
def save_model(model, prefix, folder = SAVE_FOLDER):
    model.save(f"{folder}_{prefix}_{time.strftime('%Y%m%d_%H%M%S')}")

# entraîne le modèle donné dans un environnement d'entraînement pendant n steps
def train_model_in_environment(model, timesteps=TIMESTEPS_PER_EPOCH, training_env=None):
    if training_env is None:
        training_env = create_environment(evaluation=False)

    print(f"Training model for {timesteps} steps...")
    model.set_env(training_env) # todo : tb_log_name paramètre
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=None, reset_num_timesteps=False, tb_log_name='night_run')
    training_env.close() # nécessaire de fermer l'environnement car pybullet ne peut pas avoir plusieurs environnements de simulation en parallèle (entraînement + évaluation/visualisation)

def visualize_model_in_environment(model, num_episodes=5):
    for i in range(num_episodes):
        eval_env = create_environment(evaluation=True)
        model.set_env(eval_env)

        action = np.zeros(eval_env.action_space.shape)
        START = time.time()
        terminated = False
        step = 0
        while not terminated:
            obs, reward, terminated, truncated, info = eval_env.step(action)
            action, _ = model.predict(obs)

            # synchronisation de l'affichage de la simulation
            # seulement si on veut l'interface grahique
            sync(step, START, eval_env.CTRL_TIMESTEP)
            step += 1
        eval_env.close()