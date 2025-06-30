#
# Ce fichier contient des fonctions encapsulant certaines étapes répétitives de l'apprentissage par renforcement
# en particulier, l'entraînement, la visualisation et la sauvegarde du modèle.
# Le but de ces fonctions est de pouvoir les utiliser pour facilement "scripter" un scénario en écrivant un minimum de code
#

import time
import os
from stable_baselines3 import PPO,DDPG

from reinforcement_learning_env import *
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from reinforcement_learning_env import *
from gym_pybullet_drones.utils.utils import sync

# todo : où mettre ces paramètres ?
DRONE_MODEL = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# position de départ et attitude (roll, pitch, yaw)
INIT_XYZS = np.array([[0,0,0.2]])
INIT_RPYS = np.array([np.zeros((3,))])

PHYSICS = Physics("pyb")

SAVE_FOLDER = 'models/'
TIMESTEPS_PER_EPOCH = 500

def create_environment(evaluation=False):
    environment = ReinforcementLearningEnv(drone_model=DRONE_MODEL,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                    ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                    physics=PHYSICS,
                    gui=evaluation,
                    record=DEFAULT_RECORD_VISION,
                    user_debug_gui=DEFAULT_USER_DEBUG_GUI
                    )
    return environment

# créé un nouveau modèle de RL
def create_model(env=None):
    if env is None:
        dummy_env = ReinforcementLearningEnv() # si aucun environnement n'est spécifié, créé un environnement artificiel qui est immédiatement détruit. c'est juste pour que ça compile
        model = DDPG('MlpPolicy', env=dummy_env, verbose=1)
        dummy_env.close()
        return model
    return DDPG('MlpPolicy', env=env, verbose=1)

# charge un modèle à partir d'un fichier
def load_model(filename, type, env, use_default_folder=True):
    path = (SAVE_FOLDER + filename) if use_default_folder else filename

    if os.path.isfile(filename):
        return type.load(filename, env=env)
    else:
        print(f'Error when loading model : "{filename}" doesn\'t exist.')
        return None

# charge tous les modèles avec le même préfixe dans le dossier donné
# généralement, les modèles avec le même préfixe correspondent à la même session d'entraînement
def load_models(path, prefix):
    # todo implémenter si besoin
    return None

# sauvegarde un modèle dans path et le nomme de la manière suivante : "prefix_timestamp.zip"
def save_model(model, prefix, folder = SAVE_FOLDER):
    model.save(f"{SAVE_FOLDER}ppo_model_{time.strftime('%Y%m%d_%H%M%S')}")

# entraîne le modèle donné dans un environnement d'entraînement pendant n steps
def train_model_in_environment(model, timesteps=TIMESTEPS_PER_EPOCH, training_env=None):
    if training_env is None:
        training_env = create_environment(evaluation=False)

    print(f"Training model for {timesteps} steps...")
    model.set_env(training_env)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=None)
    training_env.close() # nécessaire de fermer l'environnement car pybullet ne peut pas avoir plusieurs environnements de simulation en parallèle (entraînement + évaluation/visualisation)

def visualize_model_in_environment(model, num_episodes=5):
    for i in range(num_episodes):
        eval_env = create_environment(evaluation=True)

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