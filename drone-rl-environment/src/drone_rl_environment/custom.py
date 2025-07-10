"""
based on pid.py from gym-pybullet-drones
Lucas Charbonnier
"""

import os
import time
import timeit
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from stable_baselines3 import PPO,DDPG

from rl_utils import *

print(os.getcwd())

RUN_NAME = 'ignoremejusttesting'
STEP_PER_ITERATION = 16384

#model = load_model('ppo-nightrun2_ts-18006016_10-13-01-12.zip')
model = create_model()

# simulation
i = 0
while True:
    i += 1
    # entraînement du modèle
    train_model_in_environment(model, timesteps=STEP_PER_ITERATION, tb_run_name=RUN_NAME)

    # sauvegarde
    #save_model(model, prefix=RUN_NAME+'_ts-'+str(i * STEP_PER_ITERATION))

    # visualisation de la performance du drone dans une GUI
    visualize_model_in_environment(model)


