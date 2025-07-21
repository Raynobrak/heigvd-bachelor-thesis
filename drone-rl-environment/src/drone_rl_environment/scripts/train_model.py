# Permet d'entraîner un modèle de RL dans un environnement gym
# Exemple : poetry run python .\src\drone_rl_environment\scripts\train_model.py

from drone_rl_environment.rl_utils import *

def run():
    RUN_NAME = 'ignore-dqn-test'
    STEP_PER_ITERATION = 16384 * 16

    model = create_model()

    # simulation
    i = 0
    while True:
        i += 1
        # entraînement du modèle
        train_model_in_environment(model, timesteps=STEP_PER_ITERATION, tb_run_name=RUN_NAME)

        # sauvegarde
        save_model(model, prefix=RUN_NAME+'_ts-'+str(i * STEP_PER_ITERATION))

if __name__ == '__main__':
    run()
