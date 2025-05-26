from utils import *
from sensors.inertial_unit import *

class Drone:
    def __init__(self, initial_position = None, initial_velocity = None):
        self.position = initial_position or vec(0,0)
        self.velocity = initial_velocity or vec(0,0)

        self.inertial_unit = InertialUnit(self.position)

    def accelerate_for(self, direction, dt):
        # ensure direction is a unit vector
        direction /= direction.magnitude()

        self.velocity += direction * DRONE_ACCELERATION * dt

    def set_velocity(self, vel):
        # todo : idée -> émuler un PID pour atteindre la vitesse désirée
        self.velocity = vel

    def get_center_position(self):
        return self.position + DRONE_SIZE / 2

    def update(self, dt):
        self.position += self.velocity * dt
        
        self.inertial_unit.try_update_sensor(self.velocity, dt)

    def get_rect(self):
        return pygame.Rect(self.position.x, self.position.y, DRONE_SIZE.x, DRONE_SIZE.y)

    def collides_with(self, wall):
        return self.get_rect().colliderect(wall.rect)

    def read_accelerometer_value(self):
        return self.inertial_unit.read_sensor_acceleration()

    def display_on_window(self, surface, estimated_pos=False):
        pygame.draw.rect(surface, DRONE_COLOR, (self.position.x, self.position.y, DRONE_SIZE.x, DRONE_SIZE.y))

        if estimated_pos:
            iu_estimation = self.inertial_unit.read_sensor_estimated_position()
            pygame.draw.rect(surface, IU_PREDICTION_COLOR, (iu_estimation.x, iu_estimation.y, DRONE_SIZE.x, DRONE_SIZE.y))

    