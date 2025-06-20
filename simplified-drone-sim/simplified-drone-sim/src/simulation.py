import pygame

from src.drone import Drone

from src.constants import *
from src.wall import *
from src.sensors.lidar import *
from src.Map import *

from enum import Enum

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
        self.time_since_last_display = 0

    def reset_sim(self):
        self.drone = Drone()
        self.build_map()
        
    def build_map(self):
        self.obstacles = []
        self.obstacles.append(Wall(vec(200,200),vec(40,90)))
        self.obstacles.append(Wall(vec(350,250),vec(50,50)))

        self.obstacles.append(Wall(vec(-50,-50), vec(50, 50 + WINDOW_HEIGHT + 50)))
        self.obstacles.append(Wall(vec(-50,-50), vec(50 + WINDOW_WIDTH + 50, 50)))
        self.obstacles.append(Wall(vec(-50, WINDOW_HEIGHT), vec(50 + WINDOW_WIDTH + 50, 50)))

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

        # todo : show lidar map
        """ # todo : à enlever dès que le slam 2d fonctionne
        self.map.add_scan_at_pos(sensor_pos, data)

        map_points = self.map.get_map_points()
        print(len(map_points))
        for p in map_points:   
            pygame.draw.circle(self.window, (255,255,255), p, LIDAR_POINT_RADIUS)
        

        pixel_size = vec(WINDOW_WIDTH / MAP_SIZE, WINDOW_HEIGHT / MAP_SIZE)
        gmap = self.map.gmap
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                color = (50,10 * gridmap.GetGridProb((x,y)),50)
                pygame.draw.rect(self.window, color, pygame.Rect(x * pixel_size.x, y * pixel_size.y, pixel_size.x, pixel_size.y))
        """

        # draw accelerometer forces for visualisation
        acceleration = self.drone.read_accelerometer_value()
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(1,0) * acceleration.x * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)
        pygame.draw.line(self.window, IU_ARROWS_COLOR, sensor_pos, sensor_pos + vec(0,1) * acceleration.y * IU_ARROW_LENGTH_MULTIPLIER, IU_ARROW_WIDTH)

    def draw_slam(self):
        for wall in self.obstacles: 
            wall.display_on_window(self.window)

        self.drone.display_on_window(self.window, self.show_estimated_position)

        # draw lidar points
        sensor_pos = self.drone.get_center_position()
        lidar_distances = emulate_lidar(sensor_pos, self.obstacles)
        points = lidar_data_to_points(lidar_distances, sensor_pos)
        for p in points:
            pygame.draw.circle(self.window, LIDAR_POINT_COLOR, p, LIDAR_POINT_RADIUS)

        exact_pos = self.drone.get_center_position()
        exact_motion = exact_pos - self.last_drone_pos
        self.last_drone_pos = exact_pos

        lidar_distances = [px_to_mm(d) for d in lidar_distances]
        
        delta = px_to_mm(exact_motion)
        dxy, angle = delta.as_polar()
        motion = (dxy, 0, 1 / SIMULATION_FPS)

        estimated_pos = self.map.update_slam(motion, lidar_distances)

        print(px_to_mm(exact_pos), estimated_pos)

        slam_map = self.map.get_map()

        # todo : show lidar map
        # todo : à enlever dès que le slam 2d fonctionne

        self.time_since_last_display += 1 / SIMULATION_FPS
        if self.time_since_last_display > 5:
            self.map.display()
            self.time_since_last_display = 0