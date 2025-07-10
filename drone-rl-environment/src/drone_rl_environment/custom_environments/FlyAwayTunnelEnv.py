from .FlyAwayEnv import *

DEFAULT_TUNNEL_WIDTH = 1
DEFAULT_TUNNEL_HEIGHT = 1
DEFAULT_TUNNEL_LENGTH = 20
DEFAULT_WALLS_WIDTH = 0.1
DEFAULT_TUNNEL_EXTRA_ROOM = 0.5 # espace en plus pour éviter que le drone spawn dans un mur en [0,0,0]. c'est un décalage du tunnel vers l'arrière

TUNNEL_COLOR = [1, 1, 1, 0.5]

class FlyAwayTunnelEnv(FlyAwayEnv):
    def __init__(self,
                 tunnel_width=DEFAULT_TUNNEL_WIDTH,
                 tunnel_height=DEFAULT_TUNNEL_HEIGHT,
                 tunnel_length=DEFAULT_TUNNEL_LENGTH,
                 walls_width = DEFAULT_WALLS_WIDTH,
                 tunnel_extra_room = DEFAULT_TUNNEL_EXTRA_ROOM,
                 pybullet_physics_freq=DEFAULT_PYBULLET_PHYSICS_FREQ,
                 env_step_freq=DEFAULT_ENV_STEP_FREQ,
                 lidar_rays_count=DEFAULT_LIDAR_RAYS_COUNT,
                 lidar_max_distance=DEFAULT_LIDAR_MAX_DISTANCE,
                 lidar_freq=DEFAULT_LIDAR_FREQUENCY,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 gui=False,
                 ):
        self.tunnel_width = tunnel_width
        self.tunnel_height = tunnel_height
        self.tunnel_length = tunnel_length
        self.walls_width = walls_width
        self.default_tunnel_extra_room = tunnel_extra_room

        super().__init__(pybullet_physics_freq=pybullet_physics_freq,
                         env_step_freq=env_step_freq,
                         lidar_rays_count=lidar_rays_count,
                         lidar_max_distance=lidar_max_distance,
                         lidar_freq=lidar_freq,
                         initial_xyz_position=initial_xyz_position,
                         initial_rpy_attitude=initial_rpy_attitude,
                         max_drone_velocity=max_drone_velocity,
                         max_episode_duration=max_episode_duration,
                         gui=gui
                         )

    def _addObstacles(self):
        hw = self.walls_width/2 # moitié de la largeur des murs

        # plafond
        self.add_fixed_obstacle(
            center_pos=[-self.default_tunnel_extra_room + self.tunnel_length / 2, 0, self.tunnel_height+self.walls_width/2], 
            size=[self.tunnel_length, self.tunnel_width, self.walls_width], 
            rgba_color=TUNNEL_COLOR)

        # mur à gauche
        self.add_fixed_obstacle(
            center_pos=[-self.default_tunnel_extra_room + self.tunnel_length / 2, self.tunnel_width / 2 + hw, self.tunnel_height / 2], 
            size=[self.tunnel_length, self.walls_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR)

        # mur à droite
        self.add_fixed_obstacle(
            center_pos=[-self.default_tunnel_extra_room + self.tunnel_length / 2, -self.tunnel_width / 2 - hw, self.tunnel_height / 2], 
            size=[self.tunnel_length, self.walls_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR)

        # mur dérrière
        self.add_fixed_obstacle(
            center_pos=[-self.default_tunnel_extra_room - hw, 0, self.tunnel_height / 2], 
            size=[self.walls_width, self.tunnel_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR)