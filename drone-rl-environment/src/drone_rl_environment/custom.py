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

LOAD_MODEL = False

if LOAD_MODEL:
    env = FlyAwayCeilingEnv()
    model = load_model(r'C:\Users\lcsch\OneDrive - HESSO\Semestre6\TB\heigvd-bachelor-thesis\drone-rl-environment\models\ppo_model_20250708_103119.zip', env=FlyAwayCeilingEnv(), use_default_folder=False)
    env.close()
    if model is None:
        print('Une erreur est survenue lors du chargement du modèle.')
        exit()
else:
    model = create_model()


# simulation
while True:
    # entraînement du modèle
    train_model_in_environment(model)
    
    # sauvegarde
    #save_model(model, prefix='ppo_night_run_08-07')

    # visualisation de la performance du drone dans une GUI
    visualize_model_in_environment(model)


