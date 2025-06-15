from src.simulation import *
from src.DroneEnvironment import *

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines3 import DQN, A2C, PPO

mode = 'learn'
print('test')

if mode == 'env':
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
        n_steps=16,
        #tensorboard_log="./logs/" todo : remove
        )
    
    model = PPO(
        "MlpPolicy", env
    )
    
    while True:

        model.learn(total_timesteps=10000)

        for i in range(500000):
            env_visu = DroneEnvironment(render_mode="human")
            obs, _ = env_visu.reset()

            sum = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env_visu.step(action)
                sum += reward
                env_visu.render()
            print('total reward :', sum)

            env_visu.close()
elif mode=='learn+':
    # ======== CONFIGURATION ========
    LOAD_MODEL = True
    MODEL_PATH = "ppo_drone_model.zip"
    TRAIN_STEPS = 20000
    EVAL_EPISODES = 50000
    # ===============================

    # Entraînement : sans affichage pour aller vite
    env_train = DroneEnvironment(render_mode=None)

    # Chargement ou création du modèle
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        print("🔄 Chargement du modèle existant...")
        model = PPO.load(MODEL_PATH, env=env_train)
    else:
        print("🆕 Création d’un nouveau modèle...")
        model = PPO("MlpPolicy", env_train, verbose=1)

    while True:
        # Apprentissage
        print("🚀 Entraînement...")
        model.learn(total_timesteps=TRAIN_STEPS)

        # Sauvegarde du modèle après chaque session
        model.save(MODEL_PATH)
        print(f"💾 Modèle sauvegardé dans {MODEL_PATH}")

        # Évaluation visuelle
        print("🎮 Évaluation visuelle...")
        for episode in range(EVAL_EPISODES):
            env_eval = DroneEnvironment(render_mode="human")
            obs, _ = env_eval.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env_eval.step(action)
                if env_eval.has_user_quit():
                    break
                total_reward += reward
                env_eval.render()
            if env_eval.has_user_quit():
                break

            print(f"🏁 Reward de l’épisode {episode + 1}: {total_reward}")
            env_eval.close()