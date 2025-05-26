from simulation import *
from DroneEnvironment import *
from stable_baselines3 import DQN, A2C

import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = 'learn'

if mode == 'simu':
    sim = Simulation()
    sim.run()
elif mode == 'env':
    env = DroneEnvironment(render_mode="human")
    obs = env.reset()

    for step in range(10000):
        #action = env.action_space.sample()  # Action aléatoire
        action = vec(0,0)
        obs, reward, done, _, info = env.step(action)

        env.render()  # Affiche l’état (position, vitesse, etc.)

        if done:
            print("Épisode terminé.")
            break

    env.close()
elif mode == 'learn':
    env = DroneEnvironment(render_mode=None)

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.02,
        n_steps=16)
    model.learn(total_timesteps=50000)

    for i in range(10000):
        env_visu = DroneEnvironment(render_mode="human")
        obs, _ = env_visu.reset()

        sum = 0
        for _ in range(20000):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _, info = env_visu.step(action)
            sum += reward

            env_visu.render()
            if done:
                break
        print('total reward :', sum)

        env_visu.close()