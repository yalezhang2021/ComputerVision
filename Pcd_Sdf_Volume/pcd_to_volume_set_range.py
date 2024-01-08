# -*- coding: utf-8 -*-
"""
use the combined pcd to create volume by SDF, prepare for the next step-->marching cubes

"""
import timeit
import numpy as np
import open3d as o3d
import math
import os
from sklearn.neighbors import KDTree
from skimage import measure
from mayavi import mlab
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

input_dir = "Input/"
output_dir = "Output/"
vehicle_filename = "frame_pos_rot.txt"
vehicle_data  = np.loadtxt(input_dir + vehicle_filename, skiprows=1)#(125,7)

# vehicle position at frame 019408
vehicle_pos = np.vstack([
    -vehicle_data[0,1], #left hand coordinate to right hand  x--> -x
    vehicle_data[0,2],
    vehicle_data[0,3]]).T
print("vehicle_pos: ", vehicle_pos) #(1,3)
vehicle_pos = vehicle_pos[0]
print("vehicle_pos: ", vehicle_pos) #(1,3)
# pcd data at frame 019408
'''the point's x_axis was already x --> -x, now the points and vehicle are in the same coordinate'''
data = np.loadtxt(input_dir + "019408_point.xyz", skiprows=1) 
pcd_pos = np.array(data[:,:3])  #(n,3)
print("pcd_pos shape: ", pcd_pos.shape)

# set range[m]
max_range = 15

# set pcd range with 10m
pcd_pos_range = []
for p in pcd_pos:
    #print(p[1])
    if ((p[0]-vehicle_pos[0])**2 + (p[1]-vehicle_pos[1])**2 + (p[2]-vehicle_pos[2])**2 <= max_range**2):
        pcd_pos_range.append(p)
pcd_pos_range = np.vstack(pcd_pos_range)
print("pcd_pos_range shape: ", pcd_pos_range.shape)


#set voxel size
voxel_size = 0.05
print("voxel_size[m]:", voxel_size)

'''
now we have the pcd, and we want to fit an assembly of voxel cubes 
to approximate it. To get the voxel unit, we first need to compute 
the bounding box of the point cloud which delimit the spatial extent
of out dataset. Then we can discretize the bounding box into an assembly
of small 3D cubes: the vexels.
Note, because we need the lidar position, and each frame has different lidar
positions. for test just use a single frame.
In the future should consider how to combine many frames together
'''
start = timeit.default_timer()
class Volume:
    '''create voxel volume from 3D point cloud data'''
    def __init__(self, voxel_size, pcd_pos):
        self.voxel_size = voxel_size
        self.pcd_pos = pcd_pos_range

        x_array = self.pcd_pos[:,0]
        y_array = self.pcd_pos[:,1]
        z_array = self.pcd_pos[:,2]
        world_origin = np.array([min(x_array),min(y_array), min(z_array)])
        print("origin world coordinate: ", world_origin)
        # the box bounding should choose a small one like 10m to reduce memory and calculation
        # assign these distances to 3d voxel volume
        volume_Xaxis_length = max(x_array)-min(x_array)
        volume_Yaxis_length = max(y_array)-min(y_array)
        volume_Zaxis_length = max(z_array)-min(z_array)
        voxel_Xaxis_count = math.ceil(volume_Xaxis_length/self.voxel_size)  #get upper bound
        voxel_Yaxis_count = math.ceil(volume_Yaxis_length/self.voxel_size)
        voxel_Zaxis_count = math.ceil(volume_Zaxis_length/self.voxel_size)
        volume_box = volume_Xaxis_length*volume_Yaxis_length*volume_Zaxis_length
        voxel_count = voxel_Xaxis_count*voxel_Yaxis_count*voxel_Zaxis_count
        

        print("box volume in xyz[m]: %d x %d x %d" % (volume_Xaxis_length, volume_Yaxis_length, volume_Zaxis_length))
        print("box volume: %d" % (volume_box))
        print("voxel count: %d" % (voxel_count))
        print("Voxel size [m]: %f" % (self.voxel_size))
        

        #get voxel in Numpy array with mehgrid
        xv, yv, zv = np.meshgrid(range(voxel_Xaxis_count),range(voxel_Yaxis_count),range(voxel_Zaxis_count), indexing='ij')
        print("xv.shape:", xv.shape) #shape of the volume
        print("yv.shape:", yv.shape)
        print("zv.shape:", zv.shape)
        #get voxel cooridinate in this volume
        vox_coords = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)), axis=1)
        # may cause memory problem if choose a small voxel size like 0.02
        print("voxel_coords shape: ", vox_coords.shape) #(n,3)

        # voxel coordinates to world coordinates
        voxel_world_coords = vox_coords*voxel_size + world_origin

        #voxel center coordinates to world coordinates
        voxel_center_world_coords = voxel_world_coords + 0.5*voxel_size

        # define some parameters for the following sdf function, should rewrite in the future
        self.vehicle_pos = vehicle_pos
        self.voxel_center_world_coords = voxel_center_world_coords
        self.voxel_count = voxel_count
        self.vox_coords = vox_coords
        self.vox_world_coords = voxel_world_coords
        self.xv = xv
        
    def sdf(self):
        #find the nearest point of each voxel_center, with KD-Trees for relative large pcd
        tree = KDTree(self.pcd_pos)
        distances, indices = tree.query(self.voxel_center_world_coords, k=1)

        #project point on line(lidar<-->each voxel center) and compute the sign distance
        sign_dist = []
        for count, ind in enumerate(indices):
            nearest_point = pcd_pos[ind]
            vector_lidar_voxel = np.squeeze(self.voxel_center_world_coords[count]-self.vehicle_pos)
            vector_lidar_point = np.squeeze(nearest_point - self.vehicle_pos)
            project_point = (np.squeeze(self.vehicle_pos) + np.dot(vector_lidar_point,vector_lidar_voxel)/np.dot(
                            vector_lidar_voxel,vector_lidar_voxel)*vector_lidar_voxel)
            #print(project_point.shape) #(1,3)

            #calculate the sign distance
            dist_p = np.linalg.norm(self.vehicle_pos - project_point)
            dist_v = np.linalg.norm(self.vehicle_pos - self.voxel_center_world_coords[count])
            single_sign_dist = dist_v - dist_p
            sign_dist.append(single_sign_dist)
        sign_dist = np.vstack(sign_dist)
        print("sign dist shape: ", sign_dist.shape) #(n,1)
        sdf_volume = sign_dist.reshape(self.xv.shape) #assign these distances to volume
        print("sdf_volume.shape:", sdf_volume.shape)
        return sdf_volume
    

    # the projected point is on the plane with boundaries, the plane is nuknown, should use oriented tangend plane
    # Next, should find the undefiend sign_dist > (dense+noise) to identify the boundaries




# test class Volume
V = Volume(voxel_size, pcd_pos)
sdf_volume = V.sdf()

# marching cubes
verts, faces, normals, values = measure.marching_cubes(sdf_volume, 0)

stop = timeit.default_timer()
print("Time:", stop-start)

# # show with mlab
# mlab.triangular_mesh([vert[0] for vert in verts],
#                         [vert[1] for vert in verts],
#                         [vert[2] for vert in verts],
#                         faces)
# mlab.show()

# show in pcd and voxel grid with open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_pos)
o3d.visualization.draw_geometries([pcd])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
o3d.visualization.draw_geometries([voxel_grid])

# Get the mesh file and output it with trimesh
mesh = trimesh.Trimesh(vertices=verts,faces=faces)
mesh.show()
mesh.export(file_obj=output_dir+"sdf-mesh.ply")

mesh = trimesh.PointCloud(vertices=pcd_pos)
mesh.export(file_obj=output_dir+"point_cloud.ply")








