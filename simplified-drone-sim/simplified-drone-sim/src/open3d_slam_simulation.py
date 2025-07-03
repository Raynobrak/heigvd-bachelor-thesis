import pygame

from src.drone import Drone

from src.constants import *
from src.wall import *
from src.sensors.lidar import *

from enum import Enum

from src.slam import *

STARTING_POS = vec(100, 100)

class DisplayMode(Enum):
    NORMAL = 0,
    SENSOR_VIEW = 1,
    LIDAR_MAP_CUSTOM = 2,
    SLAM = 3

FramePerSec = pygame.time.Clock()

class SLAMSimulation:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display_mode = DisplayMode.SLAM
        self.reset_sim()

        self.last_drone_pos = self.drone.get_center_position()
        self.last_direction = vec(1,0)
        self.time_since_last_slam_update = 0

        self.slam = Slam()

    def reset_sim(self):
        self.drone = Drone()
        self.drone.position = STARTING_POS - DRONE_SIZE / 2
        self.build_map()
        
    def build_map(self):
        self.obstacles = []

        self.obstacles.append(Wall(vec(-50,-50), vec(50, 50 + WINDOW_HEIGHT + 50)))
        self.obstacles.append(Wall(vec(-50,-50), vec(50 + WINDOW_WIDTH + 50, 50)))
        self.obstacles.append(Wall(vec(-50, WINDOW_HEIGHT), vec(50 + WINDOW_WIDTH + 50, 50)))
        self.obstacles.append(Wall(vec(WINDOW_WIDTH, -50), vec(50, 50 + WINDOW_HEIGHT + 50)))

        rng = np.random.default_rng(seed=42)
        for i in range(10):
            MIN_OBST_SIZE = 40
            MAX_OBST_SIZE = 50
            pos = vec(rng.integers(150, WINDOW_WIDTH - MAX_OBST_SIZE), rng.integers(0, WINDOW_HEIGHT - MAX_OBST_SIZE))
            size = rng.integers(MIN_OBST_SIZE, MAX_OBST_SIZE)
            self.obstacles.append(Wall(pos, vec(size,size)))

    def run(self):
        pygame.display.set_caption("Game")

        while True:
            self.checkForLiveInput()
            self.checkForUserInput()

            self.update(SIMULATION_TIME_STEP)

            self.render()

            FramePerSec.tick(SIMULATION_FPS)

    def checkForUserInput(self):
        # punctal events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN :
                if event.key == pygame.K_ESCAPE:
                    self.pause_menu()
                elif event.key == pygame.K_r:
                    self.reset_sim()
                elif event.key == pygame.K_SPACE:
                    self.drone.velocity = vec(0,0)

    def checkForLiveInput(self):
        # "live" events
        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[pygame.K_w]:
            self.drone.accelerate_for(vec(0,-1), SIMULATION_TIME_STEP)
        if pressed_keys[pygame.K_a]:
            self.drone.accelerate_for(vec(-1,0), SIMULATION_TIME_STEP)
        if pressed_keys[pygame.K_s]:
            self.drone.accelerate_for(vec(0,1), SIMULATION_TIME_STEP)
        if pressed_keys[pygame.K_d]:
            self.drone.accelerate_for(vec(1,0), SIMULATION_TIME_STEP)

    def update(self, dt):
        self.drone.update(dt)
        self.update_slam(dt)

    def make_z_noise(self, vec, sigma_px: float = 0.2, rng: np.random.Generator | None = None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        z_noise = rng.normal(0.0, sigma_px)
        return float(z_noise)

    def update_slam(self, dt):
        SLAM_FREQ = 20
        self.time_since_last_slam_update += SIMULATION_TIME_STEP
        if self.time_since_last_slam_update > (1 / SLAM_FREQ):
            self.time_since_last_slam_update = 0
        else: 
            return

        # scan lidar
        lidar_distances = [d for d in emulate_lidar(self.drone.get_center_position(), self.obstacles, reverse_x=False)]
        relative_lidar_points = lidar_data_to_points(lidar_distances, vec(0,0), reverse_x=False)
        
        # relatif au référentiel
        relative_lidar_points = [vec(p.x, -p.y) for p in relative_lidar_points]

        x = [pos.x for pos in relative_lidar_points]
        y = [pos.y for pos in relative_lidar_points]
        z = [self.make_z_noise(pos) for pos in relative_lidar_points]
        column_scan = np.column_stack((x,y,z))
        self.slam.add_scan(column_scan)

        slam_pos, quat = self.slam.get_estimated_state()
        slam_pos = vec(slam_pos[0] + STARTING_POS.x, STARTING_POS.y - slam_pos[1])
        print('1. real:', self.drone.get_center_position(), '\t.2. est:', slam_pos)

        # déplacement
        displacement_mm = (self.drone.get_center_position() - self.last_drone_pos)
        self.last_drone_pos = self.drone.get_center_position()

        # changement d'angle (prograde)
        angle_delta = -self.last_direction.angle_to(self.drone.velocity)
        self.last_direction = self.drone.velocity

        # structure représentant le changement de position
        pose_change = (displacement_mm.magnitude(), angle_delta, 1 / SIMULATION_FPS)

    def render(self):
        self.window.fill((0,0,0), (0,0,self.window.get_width(), self.window.get_height()))

        self.draw_slam()

        pygame.display.update()
    
    def draw_slam(self):
        for wall in self.obstacles: 
            wall.display_on_window(self.window)

        if False:
            MAP_RAD = 2
            points, colors = self.slam.map_points_and_colors()
            for p,c in zip(points, colors):
                pygame.draw.circle(self.window, c * 255, vec(p[0], p[1]), MAP_RAD)

        self.drone.display_on_window(self.window, estimated_pos=False)

        # affichage des points lidar
        sensor_pos = self.drone.get_center_position()
        lidar_distances = emulate_lidar(sensor_pos, self.obstacles)
        points = lidar_data_to_points(lidar_distances, sensor_pos)
        for p in points:
            pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

        slam_pos, quat = self.slam.get_estimated_state()
        # pose from KISS: +X = forward, +Y = left, +Z = up
        kiss_x, kiss_y = slam_pos[0], slam_pos[1]

        # convert to screen coords: +X = right, +Y = down
        screen_x =  kiss_x            # forward  → right  (same sign)
        screen_y = -kiss_y            # left     → down   (flip sign)
        slam_pos_screen = vec(screen_x + STARTING_POS.x,
                            STARTING_POS.y + screen_y)

        rad = LIDAR_POINT_RADIUS * 2
        pygame.draw.circle(self.window, (255,0,0), slam_pos_screen, rad)