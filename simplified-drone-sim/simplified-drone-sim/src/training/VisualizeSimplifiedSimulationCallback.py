from stable_baselines3.common.callbacks import BaseCallback
from src.DroneEnvironment import DroneEnvironment

class VisualizeSimplifiedSimulationCallback(BaseCallback):
    """
    callback Stable Baselines 3 personnalisé permettant de visualiser l'environnement
    de simulation simplifié à des intervalles réguliers pendant l'entraînement.
    `visualization_env` : DroneEnvironment on which to visualize the model performances
    `visualization_freq` : Number of steps between each visualization session
    `visualization_episodes` : Number of episodes to visualize during the session
    """
    def __init__(self, visualization_freq, visualization_episodes, verbose=0):
        super(VisualizeSimplifiedSimulationCallback, self).__init__(verbose)
        self.vis_freq = visualization_freq
        self.vis_eps = visualization_episodes

        self._steps_count = 0
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        self._steps_count += 1
        if self._steps_count > self.vis_freq:
            self.visualize_episodes()
            self._steps_count = 0
        return True
    
    def visualize_episodes(self):
        print('letsgo')
        vis_env = DroneEnvironment(render_mode='human', fixed_obstacles_positions=False, seed=1)

        for episode in range(self.vis_eps):
            print(f"Visualization of episode {episode+1}/{self.vis_eps}")

            obs, _ = vis_env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = vis_env.step(action)
                if vis_env.has_user_quit():
                    break
                total_reward += reward
                vis_env.render()
            if vis_env.has_user_quit():
                break

            print(f"Total reward : {total_reward}")

        vis_env.close()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass