# -*- coding: utf-8 -*-
"""
Created on 26.01.2022
combine 20 point clouds and get the triangle mesh
@author: yao
"""
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.io as sio
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import transforms

# set dir
data_directory = ""
lidar_directory = os.path.join(data_directory, "Lidar")
image_output_directory = os.path.join(data_directory, "Lidar-Images-Transformed")
pcd_output_directory = os.path.join(data_directory,'Point-Cloud-Ply/')
mesh_output_directory = os.path.join(data_directory, 'Triangle-Mesh/')
transformation_filename = "2021-09-09-16-06-00-rundfahrt-Yi_for_yao_tf_map_odom.mat" #auto position and orientation in quat format
localization_mat = sio.loadmat(transformation_filename)

# vstack() is used to stack arrays in sequence vertically (row wise).
translation = np.vstack([
    localization_mat['translation_x'][0],
    localization_mat['translation_y'][0],
    localization_mat['translation_z'][0]]).T

rotation = np.vstack([
    localization_mat['quat_x'][0],
    localization_mat['quat_y'][0],
    localization_mat['quat_z'][0],
    localization_mat['quat_w'][0]]).T

localization_stamps = np.array(localization_mat['t'][0])   #auto time stamps


def find_matching_transformation(time_stamp):
    for idx, localization_stamp in enumerate(localization_stamps):
        if localization_stamp > time_stamp:
            return translation[idx], rotation[idx]    
    

if not os.path.exists(image_output_directory):
    os.mkdir(image_output_directory)

filenames = next(os.walk(lidar_directory))[2]               # next() next item from the iterator; walk() walking the file tree in the dir
sample_number = 20
lidar_filenames  = []
count_file = -1
for filename in filenames:
    if filename.endswith(".mat"):                           
        lidar_filenames.append(os.path.join(lidar_directory, filename))
        count_file += 1
     
lidar_filenames = sorted(lidar_filenames)                   # sorted排序
lidar_arrays = []
lidar_time_stamps = []

# sample 20 point clouds
step = int(count_file/sample_number)
for i in range(20):
    samples = i * step
    lidar_mat = sio.loadmat(lidar_filenames[samples])
    lidar_array = np.vstack((
        lidar_mat['x'], lidar_mat['y'],lidar_mat['z'],lidar_mat['intensity'])).T
    lidar_arrays.append(lidar_array)
    lidar_time_stamps.append( lidar_mat['t'][0,0])

pos_combine = []
for idx, lidar_array in enumerate(lidar_arrays): #point cloud position, intensity, and time stamps
    print(idx)
    x = lidar_array[:,0]
    y = lidar_array[:,1]
    z = lidar_array[:,2]
    i = lidar_array[:,3]
    t = lidar_time_stamps[idx]

    trans, rot = find_matching_transformation(t)
    quat = R.from_quat(rot) #auto orientation in quat

    # lidar was not 100% correctly attached, maybe apply this before
    # correction_quat = R.from_quat([0.0, 0.0, 0.06540312923, 0.99785892])
    pos = np.array([x, y, z]).T # maybe ignore z    # point position in xyz 这些点本来没有旋转，都是在一个水平坐标系里，需要让它们根据汽车方向来旋转，才能把不同点云放到一个坐标系里
    pos = quat.apply(pos)                           # change point position into quat? apply()这里相当于旋转了，这么简单吗、、、让这些点根据汽车方向旋转了
    pos += trans                                    # 旋转后再加上汽车的位置，就得到了新的位置
    #print(pos.shape)   #-->(n,3)
    pos_combine.append(pos) #-->(21,)
    
    
    # fig and save them
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if idx == 0:
        x_lim = np.ptp(pos[:,0])
        y_lim = np.ptp(pos[:,1])
        z_lim = np.ptp(pos[:,2])
    ax.set_box_aspect((x_lim, y_lim, z_lim))
    ax.set_axis_off()
    #set_axes_equal(ax)
    ax.axes.set_xlim3d(left=-100, right=100) 
    ax.axes.set_ylim3d(bottom=-100, top=100) 
    ax.axes.set_zlim3d(bottom=-12, top=20)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=0.1, c=z, cmap='winter')
    filename = os.path.join(image_output_directory, f"lidar_image_{idx:05d}.jpg")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
pos_combine = np.vstack(pos_combine)
print(pos_combine.shape)
pcd_combine = o3d.geometry.PointCloud()
#pcd points
pcd_combine.points = o3d.utility.Vector3dVector(pos_combine[:,:3])
#pcd normals
pcd_combine.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
pcd_combine.estimate_normals()

#voxel down sample
pcd_combine = pcd_combine.voxel_down_sample(voxel_size=0.02)

#quick view
o3d.visualization.draw_geometries([pcd_combine])
#save pcd_combine
o3d.io.write_point_cloud(pcd_output_directory+"pcd.ply", pcd_combine)

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

