from .FlyAwayEnv import *

DEFAULT_ENABLE_RANDOM_TUNNEL_ROTATION = False
DEFAULT_TUNNEL_WIDTH = 1
DEFAULT_TUNNEL_HEIGHT = 1
DEFAULT_TUNNEL_LENGTH = 20
DEFAULT_WALLS_WIDTH = 0.1
DEFAULT_TUNNEL_EXTRA_ROOM = 0.5 # espace en plus pour éviter que le drone spawn dans un mur en [0,0,0]. c'est un décalage du tunnel vers l'arrière

TUNNEL_COLOR = [1, 1, 1, 0.5]

class FlyAwayTunnelEnv(FlyAwayEnv):
    def __init__(self,
                 enable_random_tunnel_rotation=DEFAULT_ENABLE_RANDOM_TUNNEL_ROTATION,
                 tunnel_width=DEFAULT_TUNNEL_WIDTH,
                 tunnel_height=DEFAULT_TUNNEL_HEIGHT,
                 tunnel_length=DEFAULT_TUNNEL_LENGTH,
                 walls_width = DEFAULT_WALLS_WIDTH,
                 tunnel_extra_room = DEFAULT_TUNNEL_EXTRA_ROOM,
                 pybullet_physics_freq=DEFAULT_PYBULLET_PHYSICS_FREQ,
                 pid_controller_freq=DEFAULT_PID_CONTROLLER_FREQ,
                 action_freq=DEFAULT_ACTION_FREQ,
                 lidar_rays_count=DEFAULT_LIDAR_RAYS_COUNT,
                 lidar_max_distance=DEFAULT_LIDAR_MAX_DISTANCE,
                 lidar_freq=DEFAULT_LIDAR_FREQUENCY,
                 enable_lidar_rays_debug= DEFAULT_ENABLE_LIDAR_RAYS,
                 enable_mapping=False,
                 initial_xyz_position=None,
                 initial_rpy_attitude=None,
                 max_drone_velocity=DEFAULT_MAX_VELOCITY,
                 max_episode_duration=DEFAULT_MAX_EPISODE_DURATION,
                 drone_up_and_down_speed_multiplier=DEFAULT_DRONE_UP_AND_DOWN_SPEED_MULTIPLIER,
                 gui=False,
                 ):
        self.enable_random_tunnel_rotation = enable_random_tunnel_rotation
        self.tunnel_width = tunnel_width
        self.tunnel_height = tunnel_height
        self.tunnel_length = tunnel_length
        self.walls_width = walls_width
        self.default_tunnel_extra_room = tunnel_extra_room

        super().__init__(pybullet_physics_freq=pybullet_physics_freq,
                         pid_controller_freq=pid_controller_freq,
                         action_freq=action_freq,
                         lidar_rays_count=lidar_rays_count,
                         lidar_max_distance=lidar_max_distance,
                         lidar_freq=lidar_freq,
                         enable_lidar_rays_debug=enable_lidar_rays_debug,
                         enable_mapping=enable_mapping,
                         initial_xyz_position=initial_xyz_position,
                         initial_rpy_attitude=initial_rpy_attitude,
                         max_drone_velocity=max_drone_velocity,
                         max_episode_duration=max_episode_duration,
                         drone_up_and_down_speed_multiplier=drone_up_and_down_speed_multiplier,
                         gui=gui
                         )

    def _addObstacles(self):
        hw = self.walls_width/2 # moitié de la largeur des murs

        if self.enable_random_tunnel_rotation:
            self.tunnel_angle = self.rng.uniform(0, 2*np.pi)
            self.current_tunnel_rotation = np.array([0,0,self.tunnel_angle]) # rotation aléatoire sur l'axe 'yaw' (horizontal) entre 0 et 2pi (0 et 360°)
        else: 
            self.current_tunnel_rotation = np.array([0,0,0])

        # plafond
        self.add_fixed_obstacle(
            center_pos=self.rotate_vector_by_rpy([-self.default_tunnel_extra_room + self.tunnel_length / 2, 0, self.tunnel_height+self.walls_width/2], self.current_tunnel_rotation), 
            size=[self.tunnel_length, self.tunnel_width, self.walls_width], 
            rgba_color=TUNNEL_COLOR, 
            rotation_rpy=self.current_tunnel_rotation)

        # mur à gauche
        self.add_fixed_obstacle(
            center_pos=self.rotate_vector_by_rpy([-self.default_tunnel_extra_room + self.tunnel_length / 2, self.tunnel_width / 2 + hw, self.tunnel_height / 2], self.current_tunnel_rotation), 
            size=[self.tunnel_length, self.walls_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR, 
            rotation_rpy=self.current_tunnel_rotation)

        # mur à droite
        self.add_fixed_obstacle(
            center_pos=self.rotate_vector_by_rpy([-self.default_tunnel_extra_room + self.tunnel_length / 2, -self.tunnel_width / 2 - hw, self.tunnel_height / 2], self.current_tunnel_rotation), 
            size=[self.tunnel_length, self.walls_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR, 
            rotation_rpy=self.current_tunnel_rotation)

        # mur dérrière
        self.add_fixed_obstacle(
            center_pos=self.rotate_vector_by_rpy([-self.default_tunnel_extra_room - hw, 0, self.tunnel_height / 2], self.current_tunnel_rotation), 
            size=[self.walls_width, self.tunnel_width, self.tunnel_height], 
            rgba_color=TUNNEL_COLOR, 
            rotation_rpy=self.current_tunnel_rotation)