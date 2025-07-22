import numpy as np
from PIL import Image
from pathlib import Path

START_POSITION_COLOR = (255,0,0)
PATH_COLOR = (0,255,0)
LAST_POSITION_COLOR = (0,0,255)

MAP_SAVE_FOLDER = 'maps'

#todo refactor map

# Représente une carte en 3 dimensions
# Permet de cartographier un trajet à partir de scans globaux
class Map:
    # Construit la carte. x,y et z doivent être spécifiés en coordonnées "monde" (mètres par exemple) et resolution_voxels_per_unit correspond à la sous-division de cette unité en unités plus petites.
    # Si x,y,z = 2,2,2 et que la résolution vaut 10, alors la carte contiendra 20^3 valeurs donc chaque mètre cube est divisé en voxels de 10*10*10 centimètres
    def __init__(self, xyz_size, origin_offset, resolution_voxels_per_unit):
        self.origin_offset = origin_offset # coordonnées du point 0,0 en mètres
        self.resolution_voxels_per_unit = resolution_voxels_per_unit
        self.voxel_size = np.array([1,1,1]) / self.resolution_voxels_per_unit
        self.matrix = np.zeros(shape=resolution_voxels_per_unit * xyz_size)

        self.position_history = []
    
    @property
    def x_size(self): return self.matrix.shape[0]
    @property
    def y_size(self): return self.matrix.shape[1]
    @property
    def z_size(self): return self.matrix.shape[2]

    def update_path_history(self, coords):
        if not self.is_voxel_within_map_bounds(coords):
            return

        if len(self.position_history) > 0 :
            last_coords = self.position_history[-1]
            if (coords[:2] == last_coords[:2]).all(): # x et y différents
                return
        # on ajoute les coordonnées uniquement si x et y sont différents par rapport à la dernière position enregistrée (pour économiser un peu de mémoire en évitant de se retrouver avec une énorme liste)
        self.position_history.append(coords) 
    
    # Retourne True si les coordonnées d'un voxel donné rentrent dans la matrice
    def is_voxel_within_map_bounds(self, coords):
        return (coords[0] > 0 and coords[0] < self.x_size and
                coords[1] > 0 and coords[1] < self.y_size and
                coords[2] > 0 and coords[2] < self.z_size)
    
    # Convertir un point (en mètres) en un index à 3 dimensions représentant sa position dans les voxels de la carte
    def convert_position_to_voxel_coords(self, point):
        coords = np.floor(np.divide(point + self.origin_offset, self.voxel_size))
        return coords.astype(int)
    
    # Met à jour la carte 3D à partir d'un scan local à une position
    # Le scan est un nuage de points en 3 dimensions (liste de listes de longueur 3) -> [[x1,y1,z1], [xn,yn,zn], ...]
    def add_scan(self, local_scan_points, sensor_position, max_distance, ignore_max_distance=True):
        self.update_path_history(self.convert_position_to_voxel_coords(sensor_position))

        for p in local_scan_points:
            # si le scan correspond à la distance max du lidar, on ignore le point
            # (à condition que cette fonctionnalité soit activée)
            if (not ignore_max_distance) or (ignore_max_distance and np.linalg.norm(p) < max_distance):
                global_point_position = p + sensor_position
                voxel_coords = self.convert_position_to_voxel_coords(global_point_position)
                
                # on met à jour la carte, à condition que le point soit dans les limites
                if self.is_voxel_within_map_bounds(voxel_coords):
                    self.matrix[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = 1
    
    # "Applatit" la carte en 2 dimensions sur l'axe Z (vue d'en haut)
    # Permet d'obtenir une représentation en 2D pour visualiser sur une image p.ex.
    def flatten_map_2D(self):
        flattened = self.matrix.sum(axis=2)
        return flattened

    # Sauvegarde l'image vers un fichier
    def save_2D_map_to_file(self, suffix=None):
        print('Sauvegarde de l\'image en cours...')
        # créer une image grayscale basée sur la somme des valeurs le long de l'axe Z
        flattened = self.flatten_map_2D()
        map_2d = 255 * flattened / np.max(flattened)
        img = Image.fromarray(map_2d.astype(np.uint8))

        # conversion en RGB pour ajouter le tracé (path) du scan
        img = img.convert('RGB')
        if len(self.position_history) >= 3:
            for i, coords in enumerate(self.position_history):                
                img.putpixel((coords[1], coords[0]), PATH_COLOR)
            last = len(self.position_history) - 1
            img.putpixel((self.position_history[0][1], self.position_history[0][0]), START_POSITION_COLOR)
            img.putpixel((self.position_history[last][1], self.position_history[last][0]), LAST_POSITION_COLOR)

        path = Path.cwd() / MAP_SAVE_FOLDER / ('map'+suffix+'.png')
        img.save(path)
        print('Image sauvegardée :', path)

