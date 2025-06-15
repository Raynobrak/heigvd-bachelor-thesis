import pygame

from src.drone import Drone

from src.constants import *
from src.wall import *
from src.sensors.lidar import *

FramePerSec = pygame.time.Clock()

class Simulation:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.what_drone_sees_mode = False
        self.show_estimated_position = False
        self.reset_sim()

    def reset_sim(self):
        self.drone = Drone()
        self.build_map()
        
    def build_map(self):
        self.obstacles = []
        self.obstacles.append(Wall(vec(200,200),vec(40,90)))
        self.obstacles.append(Wall(vec(350,250),vec(50,50)))

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
                    self.what_drone_sees_mode = not self.what_drone_sees_mode
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

        if self.what_drone_sees_mode:
            self.draw_what_drone_sees()
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
