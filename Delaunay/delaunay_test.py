# -*- coding: utf-8 -*-
"""
Created on 27.01.2022
combine 20 point clouds and get the triangle mesh, the data from CARLA
@author: yao
"""

import numpy as np
from scipy.spatial import Delaunay
import trimesh

######################################
######################################
# this part is used to get all the point, here just use one frame to simplify it
lidar_data = np.loadtxt('019577_point.xyz', skiprows=1)
point_xyz = np.array(lidar_data[:,:3])  
#pos_combine_for_delaunay = pos_combine
######################################
######################################
pos_combine = point_xyz   # all points in xyz form, vertix
print(pos_combine.shape)
tri = Delaunay(pos_combine)   # Delaunay triangle --> output is an 3d object

#faces
f = tri.simplices

# Mesh
mesh = trimesh.Trimesh(vertices=pos_combine, faces=f)

# show mesh and export it
mesh.show()
mesh.export(file_obj="mesh.ply")
