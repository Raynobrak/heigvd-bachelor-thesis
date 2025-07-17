import numpy as np
from mapping.map import Map

map = Map(x_size=80, y_size=80, z_size=80, origin_offset=np.array([40,40,40]), resolution_voxels_per_unit=1) # todo : construire que si activé

for i in range(10):
    map.add_scan([[i,0,0]], [i,i,i], max_distance=100)

map.save_2D_map_to_file()
