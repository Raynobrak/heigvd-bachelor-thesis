"""
based on pid.py from gym-pybullet-drones
Lucas Charbonnier
"""

import time
import timeit
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from stable_baselines3 import PPO,DDPG

from rl_utils import *

model = create_model()

# simulation
while True:
    # entraînement du modèle
    train_model_in_environment(model)
    
    # sauvegarde
    save_model(model, prefix='DDPG')

    # visualisation de la performance du drone dans une GUI
    visualize_model_in_environment(model)


