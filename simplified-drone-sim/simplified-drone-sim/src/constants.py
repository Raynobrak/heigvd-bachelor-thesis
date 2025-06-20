import pygame
import copy

vec = pygame.math.Vector2

PIXELS_PER_METER = 50
PIXELS_PER_MILLIMETERS = PIXELS_PER_METER / 1000

def px_to_meters(px): return px / PIXELS_PER_METER
def px_to_mm(px): return px / PIXELS_PER_MILLIMETERS
def meters_to_px(meters): return meters * PIXELS_PER_METER
def mm_to_px(mm): return mm * PIXELS_PER_MILLIMETERS

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 450

SIMULATION_FPS = 60
SIMULATION_TIME_STEP = 1/SIMULATION_FPS

DRONE_SIZE = vec(40,40) # unit
DRONE_ACCELERATION = 200 # unit/s/s
DRONE_COLOR = (167, 194, 32)

# lidar sensor
NB_LIDAR_ANGLES = 64
MAX_LIDAR_DISTANCE = 1000

LIDAR_POINT_COLOR = (0,0,255)
LIDAR_POINT_RADIUS = 3

# inertial unit sensor
IU_FREQUENCY = 60 # hZ -> number of measures per second
IU_NOISE = 0.01
IU_PREDICTION_COLOR = (255,0,0)

IU_ARROW_WIDTH = 10
IU_ARROW_LENGTH_MULTIPLIER = 5
IU_ARROWS_COLOR = (100,0,0)

# 15 seconds maximum
MAX_SIMULATION_TIME = 10