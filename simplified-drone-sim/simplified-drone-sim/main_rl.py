from src.simulation import *
from src.DroneEnvironment import *
from datetime import datetime

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines3 import DQN, A2C, PPO

MODEL_FILENAME = 'models/PPO_15-06-16h22m19s' # = None to start a new model from scratch
SAVE_PREFIX = 'PPO'
TIMESTEPS_PER_EPOCH = 20000 # number of steps between each visualisation
EPISODES_PER_VIS = 5 # number of episodes to visualize after training

training_env = DroneEnvironment(render_mode=None)

model = None
if MODEL_FILENAME is not None:
    model = model = PPO.load(MODEL_FILENAME, env=training_env)
else:
    model = PPO("MlpPolicy", training_env, verbose=1)

while True:
    # entraînement
    print(f"Training model for {TIMESTEPS_PER_EPOCH} steps...")
    model.learn(total_timesteps=TIMESTEPS_PER_EPOCH, progress_bar=True)

    # sauvegarde du modèle
    filename = 'models/' + SAVE_PREFIX + '_' + datetime.now().strftime("%d-%m-%Hh%Mm%Ss")
    model.save(filename)
    print(f"Model saved : {filename}")

    # visualisation
    eval_env = DroneEnvironment(render_mode="human")

    for episode in range(EPISODES_PER_VIS):
        print(f"Visualization of episode {episode+1}/{EPISODES_PER_VIS}")

        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            if eval_env.has_user_quit():
                break
            total_reward += reward
            eval_env.render()
        if eval_env.has_user_quit():
            break

        print(f"Total reward : {total_reward}")

    eval_env.close()