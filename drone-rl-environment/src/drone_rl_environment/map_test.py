import numpy as np
from mapping.map import Map

map = Map(x_size=10, y_size=15, z_size=20, origin_offset=np.array([5,5,5]), resolution_voxels_per_unit=1) # todo : construire que si activé

map.add_scan([[0,0,0], [2,2,2]], [0,0,0], max_distance=1000)

map.save_2D_map_to_file()
