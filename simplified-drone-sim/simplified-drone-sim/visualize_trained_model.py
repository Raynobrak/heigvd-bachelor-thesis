# permet de visualiser un modèle entraîné
# exemple : poetry run python .\simplified-drone-sim\visualize_trained_model.py .\models\PPO_final_1hour_training.zip

import argparse
import multiprocessing as mp
from pathlib import Path
from stable_baselines3 import A2C, PPO, DQN
from src.DroneEnvironment import DroneEnvironment

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

def load_any_algo(zip_path: Path):
    for name, algo_cls in ALGOS.items():
        try:
            model = algo_cls.load(zip_path)
            print(f"Modèle {name.upper()} chargé")
            return model
        except Exception:
            continue
    raise RuntimeError(f"Impossible de charger {zip_path} avec {list(ALGOS.keys())}")

def run_episodes(model, n_episodes: int = 20, seed: int | None = None):
    vis_env = DroneEnvironment(render_mode="human", seed=seed)
    model.n_envs=1
    model.set_env(vis_env)

    for ep in range(n_episodes):
        obs, _ = vis_env.reset(seed=seed)
        term = trunc = False
        ep_reward = 0.0
        print(f"Episode {ep+1}/{n_episodes}")

        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = vis_env.step(action)
            ep_reward += reward
            vis_env.render()

        print(f"Reward total : {ep_reward:.2f}")

    vis_env.close()

def main():
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Visualiser un modèle SB3")
    parser.add_argument("model_zip", help="Chemin vers le .zip du modèle")
    parser.add_argument(
        "-n", "--episodes", type=int, default=5,
        help="Nombre d'épisodes à visualiser"
    )
    args = parser.parse_args()
    model = load_any_algo(Path(args.model_zip))
    run_episodes(model, n_episodes=args.episodes)

if __name__ == "__main__":
    main()
