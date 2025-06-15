from src.simulation import *
from src.DroneEnvironment import *
from datetime import datetime

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from src.training.VisualizeSimplifiedSimulationCallback import VisualizeSimplifiedSimulationCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

MODELS_FOLDER = './models/'
MODEL_FILENAME = MODELS_FOLDER + 'PPO_final_1hour_training' # = None to start a new model from scratch
#MODEL_FILENAME = None
SAVE_PREFIX = 'ppo2'
TIMESTEPS_PER_EPOCH = 1000 # number of steps between each visualisation
EPISODES_PER_VIS = 5 # number of episodes to visualize after training

checkpoint_callback = CheckpointCallback(
  save_freq=TIMESTEPS_PER_EPOCH,
  save_path=MODELS_FOLDER,
  name_prefix=SAVE_PREFIX,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

stop_training_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5
)

# wrap de l'environnement d'évaluation avec un Monitor pour que les rewards soits corrects
eval_env = Monitor(DroneEnvironment(render_mode=None))
eval_callback = EvalCallback(
    eval_freq=TIMESTEPS_PER_EPOCH,
    n_eval_episodes=EPISODES_PER_VIS,
    eval_env=eval_env,
    callback_after_eval=stop_training_callback
)

visualization_callback = VisualizeSimplifiedSimulationCallback(
    visualization_freq=TIMESTEPS_PER_EPOCH,
    visualization_episodes=EPISODES_PER_VIS
)

callbacks = CallbackList(
    callbacks=[checkpoint_callback, eval_callback, visualization_callback]
)
# todo : remettre le callback d'evaluation pour sauvegarder le meilleur modèle régulièrement

training_env = DroneEnvironment(render_mode=None)

model = None
if MODEL_FILENAME is not None:
    model = PPO.load(MODEL_FILENAME, env=training_env)
else:
    model = PPO("MlpPolicy", env=training_env, verbose=1)

while True:
    # entraînement
    print(f"Training model for {TIMESTEPS_PER_EPOCH} steps...")
    model.learn(total_timesteps=TIMESTEPS_PER_EPOCH, progress_bar=True, callback=callbacks)