import time
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R



LIDAR_DEBUG_LINES_COLOR = [1,1,1]
LIDAR_FRONT_DEBUG_LINE_COLOR = [1,1,0]
CONTACT_POINT_DEBUG_LINE_LENGTH = 0.025
CONTACT_POINT_DEBUG_LINE_WIDTH = 4


# Représente un capteur LiDAR simulé.
# le capteur LIDAR scanne les angles suivants :
# - n angles à 360° autour du drone
# - 1 angle directement en-dessous du drone
# - 1 angle directement au-dessus du drone
class LidarSensor:
    # Créé un capteur lidar
    # nb_angles spécifie le nombre d'angles totaux. Les rayons verticaux ne sont pas compris dans ce nombre.
    # P.ex. pour avoir 4 rayons horizontaux, il faudrait spécifier nb_angles=4 et le nombre de rayons total sera de 6
    # show_debug_rays permet d'activer l'affichage des rayons, à condition que la simulation tourne en mode gui.
    def __init__(self, freq, pybullet_client_id, pybullet_drone_id, nb_angles, max_distance, show_debug_rays=False):
        self.freq = freq
        self.pybullet_client_id = pybullet_client_id
        self.pybullet_drone_id = pybullet_drone_id
        self.nb_angles = nb_angles
        self.max_distance = max_distance
        self.show_debug_rays = show_debug_rays

        self.time_to_next_update = 0
        self.debug_items_ids = []

    def __del__(self):
        for id in self.debug_items_ids:
            p.removeUserDebugItem(id, physicsClientId=self.pybullet_client_id)
        self.debug_items_ids.clear()

    # inverse de la fréquence
    def time_between_updates(self):
        return 1. / self.freq

    def rays_count(self):
        return self.nb_angles + 2 # un certain nombre d'angle + les capteurs de distance en haut et en bas

    # Met à jour la visualisation en fonction du dernier scan effectué
    def update_debug_visualization(self):
        points = self.read_global_points()

        # si la liste des ids d'objets debug n'est pas vide, on ne doit pas les recréer mais simplement les mettre à jour
        lines_exist = len(self.debug_items_ids) > 0

        for i, pt in enumerate(points):
            line_color = LIDAR_FRONT_DEBUG_LINE_COLOR if i == 0 else LIDAR_DEBUG_LINES_COLOR

            # si les lignes n'existe pas, on doit d'abord les créer
            if lines_exist: 
                line_id = self.debug_items_ids[i*2]
                id = p.addUserDebugLine(self.drone_pos, pt, line_color, replaceItemUniqueId=line_id)
                assert(id == line_id)
                
                # accentue le point d'arrivée en dessinant une ligne épaisse de très petite longueur
                # note : il existe une fonction p.addUserDebugPoints() mais il y a un bug qui fait qu'on ne peut pas supprimer ou obtenir l'ID des points ajoutés
                # d'où l'utilisation de ce "workaround" pour pouvoir ajouter un marqueur au point d'arrivée.
                contact_line_id = self.debug_items_ids[i*2+1]
                id = p.addUserDebugLine(pt - [0,0,CONTACT_POINT_DEBUG_LINE_LENGTH], pt + [0,0,CONTACT_POINT_DEBUG_LINE_LENGTH], [255, 0, 0], lineWidth=CONTACT_POINT_DEBUG_LINE_WIDTH, replaceItemUniqueId=contact_line_id)  # 0 = persistant
                assert(id == contact_line_id)
            else:
                line_id = p.addUserDebugLine(self.drone_pos, pt, line_color)

                assert(line_id >= 0)
                self.debug_items_ids.append(line_id)

                # accentue le point d'arrivée en dessinant une ligne épaisse de très petite longueur
                # note : il existe une fonction p.addUserDebugPoints() mais il y a un bug qui fait qu'on ne peut pas supprimer ou obtenir l'ID des points ajoutés
                # d'où l'utilisation de ce "workaround" pour pouvoir ajouter un marqueur au point d'arrivée.
                contact_line_id = p.addUserDebugLine(pt - [0,0,CONTACT_POINT_DEBUG_LINE_LENGTH], pt + [0,0,CONTACT_POINT_DEBUG_LINE_LENGTH], [255, 0, 0], lineWidth=CONTACT_POINT_DEBUG_LINE_WIDTH)  # 0 = persistant
                assert(contact_line_id >= 0)
                self.debug_items_ids.append(contact_line_id)
    
    # met à jour le capteur en fonction du temps écoulé depuis la dernière mise à jour
    # le temps écoulé est accumulé et comparé à la fréquence de rafraîchissement afin de ne pas mettre à jour le capteur à une fréquence plus haute que celle-ci.
    def update(self, elapsed_time):
        self.time_to_next_update -= elapsed_time
        if self.time_to_next_update < 0:
            self.time_to_next_update = self.time_between_updates()
            self._compute_internal_data()
            if self.show_debug_rays:
                self.update_debug_visualization()
    
    # calcule les distances LiDAR internes. à n'appeler que lorsque la fréquence de rafraîchissement est atteinte
    def _compute_internal_data(self):
        self.lidar_data = []
        
        # obtention de l'état du drone
        self.drone_pos, orientation_quat = p.getBasePositionAndOrientation(self.pybullet_drone_id, self.pybullet_client_id)

        # obtention d'une matrice de rotation à partir du quaternion
        rot_matrix = R.from_quat(orientation_quat).as_matrix()

        # calcul des directions de tous les angles
        rays_directions = []
        for i in range(self.nb_angles):
            # angle et direction dans le repère local du drone (plan horizontal)
            horizontal_angle_rad = (2 * np.pi * i) / self.nb_angles
            local_dir = np.array([np.cos(horizontal_angle_rad), np.sin(horizontal_angle_rad), 0])

            # direction réelle prenant en compte l'attitude (orientation) du drone
            real_dir = rot_matrix @ local_dir
            rays_directions.append(real_dir)

        # rayons supplémentaires qui pointent en haut et en bas
        rays_directions.append(rot_matrix @ np.array([0,0,1])) # vers le haut
        rays_directions.append(rot_matrix @ np.array([0,0,-1])) # vers le bas =~ altitude

        # utilisation de la fonction rayTestBatch de pybullet pour les rayons lidar
        ray_starts = [self.drone_pos for _ in range(len(rays_directions))]
        ray_ends = [self.drone_pos + rays_directions[i] * self.max_distance for i in range(len(rays_directions))]
        results = p.rayTestBatch(ray_starts, ray_ends)
        for i, r in enumerate(results):
            hit = r[0] != -1
            hit_fraction = r[2]
            distance = hit_fraction * self.max_distance if hit else self.max_distance
            
            # ajout de la donnée pour un angle : un tuple (direction,distance), la direction est un vecteur 3D
            self.lidar_data.append((rays_directions[i], distance))
    
    # retourne une liste contenant toutes les distances
    # NOTE : si cette fonction est appelé deux fois durant un intervalle de temps inférieur à la fréquence de rafraîchissement du capteur, les valeurs retournées seront les mêmes.
    def read_distances(self):
        directions, distances = zip(*self.lidar_data)
        return np.array(distances)
    
    # comme read_distances() mais les distances sont normalisées entre 0 et 1 en fonction de la distance max du capteur
    def read_normalized_distances(self):
        return self.read_distances() / self.max_distance
    
    # retourne une liste contenant tous les points locaux (référentiel = position du drone) détectés par le capteur LiDAR
    # si ignore_max_distances est True, les points correspondants à la distance max du capteur ne seront pas ajoutés à la liste
    def read_local_points(self, ignore_max_distances=False):
        return [dir * dist for dir, dist in self.lidar_data]
    
    # retourne une liste contenant tous les points locaux (référentiel = [0,0,0]) détectés par le capteur LiDAR
    # si ignore_max_distances est True, les points correspondants à la distance max du capteur ne seront pas ajoutés à la liste
    def read_global_points(self, ignore_max_distances=False):
        return [self.drone_pos + local_point for local_point in self.read_local_points()]