# -*- coding: utf-8 -*-
"""
Created on 27.01.2022
combine 20 point clouds and get the triangle mesh, the data from CARLA
@author: yao
"""

from math import degrees
from matplotlib import transforms
import scipy.io as sio
import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

input_dir = "Input-Data/"
output_dir = "Output-Data/"
vehicle_filename = "frame_pos_rot.txt"
vehicle_data  = np.loadtxt(input_dir + vehicle_filename, skiprows=1)
print(vehicle_data.shape) #(125,7)
#print(vehicle_data)

location = np.vstack([
    -vehicle_data[:,1], #left hand coordinate to right hand  x--> -x
    vehicle_data[:,2],
    vehicle_data[:,3]]).T
print(location.shape)   #(125,3)

# degrees, x,y,z, also need to change to right hand coordinate
rotation = np.vstack([
    vehicle_data[:,4],
    -vehicle_data[:,5],
    -vehicle_data[:,6]]).T
print(rotation.shape)   #(125,3)

frames = np.array(vehicle_data[:,0])
print(frames.shape)     #(125,)

filenames = next(os.walk(input_dir))[2]
number_measurements = 10
lidar_filenames  = []
for filename in filenames:
    if filename.endswith(".xyz"):                           
        lidar_filenames.append(os.path.join(input_dir, filename))

total_number = len(lidar_filenames)
        
lidar_filenames = sorted(lidar_filenames)                   # sorted排序
lidar_arrays = []
step = int(total_number/number_measurements)
print(step)

pos_combine = []
# sample 20 point clouds
for i in range(number_measurements):
    samples = i * step
    lidar_data = np.loadtxt(lidar_filenames[samples], skiprows=1)
    print(i)
    trans, rot = location[samples], rotation[samples]  #
    r = R.from_euler('xyz', rot, degrees=True) #auto orientation in euler
    length = len(lidar_data[:,0])
    for n in range(length):
        # x = -lidar_data[n,0]
        # y = lidar_data[n,1]
        # z = lidar_data[n,2]
        # point_xyz = np.array([x,y,z]).T
        point_xyz = np.array(lidar_data[n,:3]).T    # I have inversed x-axis in collecting data
        pos = r.apply(point_xyz)
        pos += trans
        pos_combine.append(pos)

pos_combine = np.vstack(pos_combine)
print(pos_combine.shape)
pcd_combine = o3d.geometry.PointCloud()
#pcd points
pcd_combine.points = o3d.utility.Vector3dVector(pos_combine[:,:3])


#quick view without voxel down sample or normals
#o3d.visualization.draw_geometries([pcd_combine])

#pcd normals estimate
pcd_combine.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
pcd_combine.estimate_normals()

#voxel down sample
pcd_combine = pcd_combine.voxel_down_sample(voxel_size=0.02)

#quick view with voxel down sample
o3d.visualization.draw_geometries([pcd_combine])
#save pcd_combine
o3d.io.write_point_cloud(output_dir+"pcd_combine.ply", pcd_combine)

#Here can use Meshlab see the mesh and estimate the ball radius, less time..

'''
#Strategy1:BPA(ball-Pivoting Algorithm)
#first compute the necessary radius parameter based on the 
#average distances computed from all the distances between points:

distances = pcd_combine.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius =[2*avg_dist, 8*avg_dist, 16*avg_dist, 24*avg_dist, 32*avg_dist, 40*avg_dist, 48*avg_dist]

#pcd.normals = o3d.geometry.TriangleMesh.compute_triangle_normals(pcd.points)
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_combine, o3d.utility
.DoubleVector(radius))  #create mesh and store it  in bpa_mesh

#downsample the result to an acceptable number of triangles, exp.100k
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

#Additionally, if you think the mesh can present some weird artifacts, you can 
#run the following commands to ensure its consistency:
# dec_mesh.remove_degenerate_triangles()
# dec_mesh.remove_duplicated_triangles() #去掉重复三角
# dec_mesh.remove_duplicated_vertices()  #去掉重复顶点
# dec_mesh.remove_non_manifold_edges()

# #Startegy2:Possion'reconstruction
# #computing the mesh
# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_combine, depth=9, width=0, scale=1.1, linear_fit=False)[0]

# #cropping
# bbox = pcd_combine.get_axis_aligned_bounding_box()
# p_mesh_crop = poisson_mesh.crop(bbox)


#output
o3d.io.write_triangle_mesh(mesh_output_directory+"bpa_mesh.ply", dec_mesh)
#o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)

#function creation
def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods = {}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i] = mesh_lod
        print("generation of "+str(i)+" LoD successful")
    return mesh_lods

#execution of function
my_lods = lod_mesh_export(dec_mesh, [10000,3000, 1000,100], ".ply", mesh_output_directory)
#my_lods = lod_mesh_export(p_mesh_crop, [100000,10000,3000, 1000,100], ".ply", output_path)

#visualize within python a specific LoD
o3d.visualization.draw_geometries([my_lods[1000]])

'''



