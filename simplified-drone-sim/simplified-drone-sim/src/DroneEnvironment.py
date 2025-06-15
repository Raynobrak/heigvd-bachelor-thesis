import gymnasium as gym
from gymnasium import spaces

import numpy as np

from src.constants import *
from src.sensors.lidar import *
from src.drone import Drone
from src.wall import *

HUMAN_RENDERMODE = 'human'

class DroneEnvironment(gym.Env):
    def __init__(self, render_mode):
        super().__init__()

        # observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=MAX_LIDAR_DISTANCE,
            shape=(NB_LIDAR_ANGLES,),  # 8 distances
            dtype=np.float32,
        )

        # action space
        self.action_space = spaces.Box(
            low=-150.0,
            high=150.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        if self.render_mode == HUMAN_RENDERMODE:
            self.fps = pygame.time.Clock()

            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.drone = Drone(vec((DRONE_SIZE.x + 100) / 2, (WINDOW_HEIGHT + DRONE_SIZE.y) / 2))

        # Réinitialiser la map, obstacles, etc.
        self._reset_environment()

        self.terminate = False
        self.elapsed_time = 0
        self.quit = False

        return self._get_observation(), {}

    def step(self, action):
        vel = vec(*action) * 50
        self.drone.set_velocity(vel)

        self.drone.update(SIMULATION_TIME_STEP)

        self.elapsed_time += SIMULATION_TIME_STEP

        if self.render_mode == HUMAN_RENDERMODE:
            self.fps.tick(SIMULATION_FPS)

            # punctal events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.quit = True

        # observation et reward pour l'état actuel
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        info = {}

        return obs, reward, done, False, info

    def _get_observation(self):
        return self._simulate_lidar()

    def _simulate_lidar(self):
        data = emulate_lidar(self.drone.get_center_position(), self.walls)
        return np.array(data)

    def _compute_reward(self):
        # reward negatif si le drone tape un obstacle
        for w in self.walls:
            if self.drone.collides_with(w):
                self.terminate = True
                return -200
            
        distance_travelled = self.drone.get_rect().right

        """
        if distance_travelled > WINDOW_WIDTH:
            self.terminate = True
            # reward pour avoir atteint la ligne d'arrivée
            # plus le drone est arrivé vite, plus le reward est grand
            return 400 / self.elapsed_time
        """

        #return -1 #todo find better reward fctn
        return 100 * distance_travelled / (WINDOW_WIDTH - DRONE_SIZE.x)

    def _check_termination(self):
        return self.terminate or self.elapsed_time > MAX_SIMULATION_TIME

    def _reset_environment(self):
        self.walls = []

        self.walls.append(Wall(vec(-50,-50), vec(50, 50 + WINDOW_HEIGHT + 50)))
        self.walls.append(Wall(vec(-50,-50), vec(50 + WINDOW_WIDTH + 50, 50)))
        self.walls.append(Wall(vec(-50, WINDOW_HEIGHT), vec(50 + WINDOW_WIDTH + 50, 50)))

        for i in range(10):
            MIN_OBST_SIZE = 40
            MAX_OBST_SIZE = 50
            pos = vec(np.random.randint(150, WINDOW_WIDTH - MAX_OBST_SIZE), np.random.randint(0, WINDOW_HEIGHT - MAX_OBST_SIZE))
            size = np.random.randint(MIN_OBST_SIZE, MAX_OBST_SIZE)
            self.walls.append(Wall(pos, vec(size,size)))

    def has_user_quit(self):
        return self.quit

    def render(self):
        if self.render_mode == HUMAN_RENDERMODE:
            self.window.fill((0,0,0), (0,0,self.window.get_width(), self.window.get_height()))

            self.drone.display_on_window(self.window, estimated_pos=False)

            for wall in self.walls: 
                wall.display_on_window(self.window)

            # emulate and draw lidar points
            sensor_pos = self.drone.get_center_position()
            data = emulate_lidar(sensor_pos, self.walls)
            points = lidar_data_to_points(data, sensor_pos)
            for p in points:
                pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

            pygame.display.update()
    
