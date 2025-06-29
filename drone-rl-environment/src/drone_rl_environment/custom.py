"""
based on pid.py from gym-pybullet-drones
Lucas Charbonnier
"""

import time
import argparse
import numpy as np
import pybullet as p
import random

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from stable_baselines3 import PPO

from reinforcement_learning_env import *

DRONE_MODEL = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

PHYSICS = Physics("pyb")

MAX_DURATION = 100
    
# start position and attitude (orientation)
INIT_XYZS = np.array([[0,0,0.2]])
INIT_RPYS = np.array([np.zeros((3,))])

# todo next time
# 1. create learning env
# 2. learn for x steps
# 3. create visualisation env to visualize model

training_env = ReinforcementLearningEnv(drone_model=DRONE_MODEL,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                    ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                    physics=PHYSICS,
                    gui=False,
                    record=DEFAULT_RECORD_VISION,
                    user_debug_gui=DEFAULT_USER_DEBUG_GUI
                    )

model = PPO("MlpPolicy", env=training_env, verbose=1)

TIMESTEPS_PER_EPOCH = 1000

# environnement


# simulation
while True:
    print(f"Training model for {TIMESTEPS_PER_EPOCH} steps...")
    #model.learn(total_timesteps=TIMESTEPS_PER_EPOCH, progress_bar=True, callback=None)

    eval_env = ReinforcementLearningEnv(drone_model=DRONE_MODEL,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                    ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                    physics=PHYSICS,
                    gui=True,
                    record=DEFAULT_RECORD_VISION,
                    user_debug_gui=DEFAULT_USER_DEBUG_GUI
                    )

    action = np.zeros((3,1))
    START = time.time()
    terminated = False
    step = 0
    while not terminated:
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated:
            exit()


        action, _ = model.predict(obs)

        #target_vel = np.array([0.2,0.2,0])
        #action = np.array(target_vel)

        # affichage de debug
        #env.render() # todo : remettre le rendering/logging

        # synchronisation de l'affichage de la simulation
        # seulement si on veut l'interface grahique
        step += 1
        sync(step, START, eval_env.CTRL_TIMESTEP)
    eval_env.close()
