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

def run():
    RUN_NAME = 'dqn-long-episodes-no-lateral-movement-128-128'
    STEP_PER_ITERATION = 16384 * 16

    #model = load_model('ppo-tunnel-nightrun_ts-21037056_11-11-50-02.zip')
    model = create_model()

    # simulation
    i = 0
    while True:
        i += 1
        # entraînement du modèle
        train_model_in_environment(model, timesteps=STEP_PER_ITERATION, tb_run_name=RUN_NAME)

        # sauvegarde
        save_model(model, prefix=RUN_NAME+'_ts-'+str(i * STEP_PER_ITERATION))

        # visualisation de la performance du drone dans une GUI
        #visualize_model_in_environment(model, num_episodes=10)

if __name__ == '__main__':
    run()
