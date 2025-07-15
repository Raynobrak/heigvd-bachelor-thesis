import numpy as np
from PIL import Image

# Représente une carte en 3 dimensions
# Permet de cartographier un trajet à partir de scans globaux
class Map:
    # Construit la carte. x,y et z doivent être spécifiés en coordonnées "monde" (mètres par exemple) et resolution_voxels_per_unit correspond à la sous-division de cette unité en unités plus petites.
    # Si x,y,z = 2,2,2 et que la résolution vaut 10, alors la carte contiendra 20^3 valeurs donc chaque mètre cube est divisé en voxels de 10*10*10 centimètres
    def __init__(self, x_size, y_size, z_size, origin_offset, resolution_voxels_per_unit):
        self.origin_offset = origin_offset # coordonnées du point 0,0 en mètres
        self.resolution_voxels_per_unit = resolution_voxels_per_unit
        self.voxel_size = np.array([1,1,1]) / self.resolution_voxels_per_unit
        self.matrix = np.zeros(shape=resolution_voxels_per_unit * np.array([x_size, y_size, z_size]))
    
    def x_size(self):
        return self.matrix.shape[0]
    def y_size(self):
        return self.matrix.shape[1]
    def z_size(self):
        return self.matrix.shape[2]
    
    def is_within_map_bounds(self, position):
        return True
        position += self.origin_offset
        return (position[0] > 0 and position[0] < self.x_size() and
                position[1] > 0 and position[1] < self.y_size() and
                position[2] > 0 and position[2] < self.z_size())
    
    # Convertir un point (en mètres) en un index à 3 dimensions représentant sa position dans les voxels de la carte
    def convert_position_to_voxel_coords(self, point):
        point += self.origin_offset
        coords = np.floor(np.divide(point, self.voxel_size))
        return coords.astype(int)
    
    # Met à jour la carte 3D à partir d'un scan local à une position
    # Le scan est un nuage de points en 3 dimensions (liste de listes de longueur 3) -> [[x1,y1,z1], [xn,yn,zn], ...]
    def add_scan(self, local_scan_points, sensor_position, max_distance, ignore_max_distance=True):
        for p in local_scan_points:
            # si le scan correspond à la distance max du lidar, on ignore le point
            if ignore_max_distance and self.is_within_map_bounds(p) and np.linalg.norm(p) < max_distance:
                global_pos = np.array(p) + np.array(sensor_position)

                # si le point est dans les limites de la carte
                if self.is_within_map_bounds(global_pos):
                    map_index = self.convert_position_to_voxel_coords(global_pos)
                    self.matrix[map_index[0], map_index[1], map_index[2]] = 1
    
    # "Applatit" la carte en 2 dimensions sur l'axe Z (vue d'en haut)
    # Permet d'obtenir une représentation en 2D pour visualiser sur une image p.ex.
    def flatten_map_2D(self):
        map = np.zeros(shape=(self.x_size(), self.y_size()))
        flattened = self.matrix.sum(axis=2)
        return flattened

    def save_2D_map_to_file(self):
        map_2d = self.flatten_map_2D()
        max_value = np.max(map_2d)
        map_2d = map_2d / max_value * 255 # normalisation des valeurs de la carte entre 0 et 1

        img_u8 = map_2d.astype(np.uint8)

        img = Image.fromarray(img_u8, mode='L')
        img.save('map-gray.png')
        print('Image sauvegardée.')

