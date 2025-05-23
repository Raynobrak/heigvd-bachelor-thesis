from utils import *

class Wall:
    def __init__(self, position, size, color = (100,100,50)):
        self.rect = pygame.Rect(position.x, position.y, size.x, size.y)
        self.color = color

    def contains_point(self, vector):
        return self.rect.collidepoint(vector)

    def intersects_obstacle(self, other):
        return self.rect.colliderect(other.rect)

    def display_on_window(self, surface):
        pygame.draw.rect(surface, self.color, (self.rect.left, self.rect.top, self.rect.width, self.rect.height))
