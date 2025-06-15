from src.constants import *
import numpy as np

#todo fix methods names that are not very self-explanatory
class InertialUnit:
    def __init__(self, starting_position):
        self.last_velocity = vec(0,0)
        self.initial_pos = starting_position
        self.estimated_position = vec(self.initial_pos)
        self.estimated_velocity = vec(0,0)
        self.last_recorded_acceleration = vec(0,0)
        self.time_since_last_update = 0

    def try_update_sensor(self, current_velocity, elapsed_time):
        self.time_since_last_update += elapsed_time
        if self.time_since_last_update > (1/ IU_FREQUENCY):
            self.time_since_last_update -= (1 / IU_FREQUENCY)
            self.force_update_sensor(current_velocity)

    def read_acceleration(self, current_velocity):
        raw_acceleration = current_velocity - self.last_velocity
        noisy_acceleration = raw_acceleration + vec(np.random.normal(0, IU_NOISE, size=1), np.random.normal(0, IU_NOISE, size=1))
        self.last_velocity = copy.copy(current_velocity)
        self.last_recorded_acceleration = copy.copy(noisy_acceleration)
        return noisy_acceleration

    def force_update_sensor(self, current_velocity):
        acc = self.read_acceleration(current_velocity)

        self.estimated_position += self.estimated_velocity * (1 / IU_FREQUENCY)
        self.estimated_velocity += acc

    def read_sensor_acceleration(self):
        return self.last_recorded_acceleration
    
    def read_sensor_estimated_position(self):
        return self.estimated_position