import pygame

from src.drone import Drone

from src.constants import *
from src.wall import *
from src.sensors.lidar import *
from src.Map import *

from enum import Enum

STARTING_POS = vec(100, 100)

class DisplayMode(Enum):
    NORMAL = 0,
    SENSOR_VIEW = 1,
    LIDAR_MAP_CUSTOM = 2,
    SLAM = 3

FramePerSec = pygame.time.Clock()

class Simulation:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display_mode = DisplayMode.NORMAL
        self.show_estimated_position = False
        self.map = Map()
        self.reset_sim()

        self.last_drone_pos = self.drone.get_center_position()
        self.last_direction = vec(1,0)
        self.time_since_last_display = 0

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
                elif event.key == pygame.K_1:
                    self.display_mode = DisplayMode.NORMAL
                elif event.key == pygame.K_2:
                    self.display_mode = DisplayMode.SENSOR_VIEW
                elif event.key == pygame.K_3:
                    self.display_mode = DisplayMode.LIDAR_MAP_CUSTOM
                elif event.key == pygame.K_4:
                    self.display_mode = DisplayMode.SLAM
                elif event.key == pygame.K_p:
                    self.show_estimated_position = not self.show_estimated_position

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
        self.check_for_collisions(dt)
        self.apply_control_strategy(dt)
        self.drone.update(dt)

        if(self.display_mode == DisplayMode.SLAM):
            self.update_slam(dt)

    def update_slam(self, dt):
        # scan lidar
        lidar_scan_mm = [px_to_mm(d) for d in emulate_lidar(self.drone.get_center_position(), self.obstacles)]

        # déplacement
        displacement_mm = px_to_mm((self.drone.get_center_position() - self.last_drone_pos))
        self.last_drone_pos = self.drone.get_center_position()

        # changement d'angle (prograde)
        angle_delta = -self.last_direction.angle_to(self.drone.velocity)
        self.last_direction = self.drone.velocity

        # structure représentant le changement de position
        pose_change = (displacement_mm.magnitude(), angle_delta, 1 / SIMULATION_FPS)

        #todo : enlever
        self.map.window_to_map(self.drone.get_center_position() - STARTING_POS)

        # màj du SLAM
        self.map.update_slam(lidar_scan_millimeters=lidar_scan_mm, motion_estimation_millimeters=pose_change)

    def get_sensor_data(self):
        return emulate_lidar(self.drone.get_center_position(), self.obstacles), self.drone.read_accelerometer_value()

    def apply_control_strategy(self, dt):
        self.checkForLiveInput()
        # idées de tâches :
        # - atteindre une position
        # - atteindre une vitesse et s'arrêter le plus vite possible
        # - atteindre une position avec des obstacles

    def check_for_collisions(self, dt):
        for wall in self.obstacles:
            # todo check collisions
            pass

    def render(self):
        self.window.fill((0,0,0), (0,0,self.window.get_width(), self.window.get_height()))

        if self.display_mode == DisplayMode.SENSOR_VIEW:
            self.draw_what_drone_sees()
        elif self.display_mode == DisplayMode.LIDAR_MAP_CUSTOM:
            self.draw_lidar_map()
        elif self.display_mode == DisplayMode.SLAM:
            self.draw_slam()
        else:
            self.draw_simulation()

        pygame.display.update()
    
    def draw_simulation(self):
        self.drone.display_on_window(self.window, self.show_estimated_position)

        for wall in self.obstacles: 
            wall.display_on_window(self.window)

        # emulate and draw lidar points
        sensor_pos = self.drone.get_center_position()
        data = emulate_lidar(sensor_pos, self.obstacles)
        points = lidar_data_to_points(data, sensor_pos)
        for p in points:
            pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

        # draw accelerometer forces for visualisation
        acceleration = self.drone.read_accelerometer_value()
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(1,0) * acceleration.x * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(0,1) * acceleration.y * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)

    def draw_what_drone_sees(self):
        window_center = vec(self.window.get_width(), self.window.get_height()) / 2

        # emulate and draw lidar points
        sensor_pos = self.drone.get_center_position()
        data = emulate_lidar(sensor_pos, self.obstacles)
        points = lidar_data_to_points(data, window_center)
        for p in points:
            pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

        # draw accelerometer forces for visualisation
        acceleration = self.drone.read_accelerometer_value()
        pygame.draw.line(self.window, IU_ARROWS_COLOR, window_center, window_center + vec(1,0) * acceleration.x * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)
        pygame.draw.line(self.window, IU_ARROWS_COLOR, window_center, window_center + vec(0,1) * acceleration.y * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)

    def draw_lidar_map(self):
        self.drone.display_on_window(self.window, self.show_estimated_position)

        # emulate and draw lidar points
        sensor_pos = self.drone.get_center_position()
        lidar_distances = emulate_lidar(sensor_pos, self.obstacles)
        points = lidar_data_to_points(lidar_distances, sensor_pos)
        for p in points:
            pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

        # draw accelerometer forces for visualisation
        acceleration = self.drone.read_accelerometer_value()
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(1,0) * acceleration.x * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(0,1) * acceleration.y * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)

    def draw_slam(self):
        for wall in self.obstacles: 
            wall.display_on_window(self.window)

        self.drone.display_on_window(self.window, self.show_estimated_position)


        # draw lidar points
        slam_pos = self.map.get_estimated_position_in_window_px(window_size_px=WINDOW_SIZE, starting_position_px=STARTING_POS)
        print(self.drone.get_center_position(), slam_pos)

        rad = LIDAR_POINT_RADIUS * 2
        pygame.draw.circle(self.window, (255,0,0), slam_pos + vec(1,1) * rad/2, rad)

        self.time_since_last_display += 1 / SIMULATION_FPS
        if self.time_since_last_display > 5:
            self.map.display()
            self.time_since_last_display = 0