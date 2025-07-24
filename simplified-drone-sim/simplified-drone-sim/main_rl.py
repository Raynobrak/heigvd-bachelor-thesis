import os
import multiprocessing
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.DroneEnvironment import DroneEnvironment
from src.training.VisualizeSimplifiedSimulationCallback import (
    VisualizeSimplifiedSimulationCallback,
)

MODELS_FOLDER = "./models/"
MODEL_FILENAME = None          # = None pour un modèle neuf
SAVE_PREFIX = "ppo"
TIMESTEPS_PER_EPOCH = 16384 * 8
EPISODES_PER_EVAL = 5
EVAL_FREQ = TIMESTEPS_PER_EPOCH / 8
EVAL_SEED = 1234
TENSORBOARD_LOGS_FOLDER = "./tensorboard-logs/"
RUN_NAME = 'a2c'
N_ENVS = 10

def make_drone_env():
    return DroneEnvironment(render_mode=None)

def main():
    training_env = make_vec_env(
        make_drone_env,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        monitor_dir="./logs",
    )

    eval_env = Monitor(DroneEnvironment(render_mode=None, seed=EVAL_SEED))

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS_PER_EPOCH,
        save_path=MODELS_FOLDER,
        name_prefix=RUN_NAME,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    stop_training_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EPISODES_PER_EVAL,
        callback_after_eval=stop_training_callback,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    if MODEL_FILENAME is not None:
        #model = PPO.load(MODEL_FILENAME, env=training_env, tensorboard_log=TENSORBOARD_LOGS_FOLDER)
        print('not implemented')
    else:
        #model = PPO("MlpPolicy", env=training_env, verbose=1, tensorboard_log=TENSORBOARD_LOGS_FOLDER, batch_size=256, clip_range=0.1, learning_rate=1e-4, policy_kwargs=dict(net_arch=[64, 32]))
        model = A2C("MlpPolicy", env=training_env, verbose=1, tensorboard_log=TENSORBOARD_LOGS_FOLDER, learning_rate=1e-4, policy_kwargs=dict(net_arch=[64, 32]))
    while True:
        print(f"Training for {TIMESTEPS_PER_EPOCH:,} steps on {N_ENVS} envs…")
        model.learn(total_timesteps=TIMESTEPS_PER_EPOCH, progress_bar=True, callback=callbacks, tb_log_name=RUN_NAME, reset_num_timesteps=False)

if __name__ == '__main__':
    main()