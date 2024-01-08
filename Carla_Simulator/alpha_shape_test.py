# -*- coding: utf-8 -*-
"""
Created on 27.01.2022
combine 20 point clouds and get the triangle mesh, the data from CARLA
@author: yao
"""

import numpy as np
import open3d as o3d
from matplotlib import transforms
import scipy.io as sio
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import trimesh


######################################
######################################
# this part is used to get all the point
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
number_measurements = 20    #sample times
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
# sample 10 point clouds
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
######################################
######################################
#pos_combine = point_xyz   # all points in xyz form, vertix
pos_combine = np.vstack(pos_combine)
print(pos_combine.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pos_combine[:,:3])
# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
for alpha in np.logspace(np.log10(1.2), np.log10(0.3), num=4):
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

