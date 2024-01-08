# -*- coding: utf-8 -*-
"""
Created on 19.01.2022

@author: yao
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import open3d as o3d


point_cloud = scipy.io.loadmat('2021-09-09-16-06-00-rundfahrt-Yi_00000_lidar.mat')
#The result is a dictionary, one key/value pair for each variable

x = point_cloud['x']                    #
y = point_cloud['y']                    #
z = point_cloud['z']                    #
intensity = point_cloud['intensity']    #
t = point_cloud['t']                    #time

# print('size of x: ', np.size(x) ,'\n'
#       'size of y: ', np.size(y) , '\n'
#       'size of z: ', np.size(z) , '\n'
#       'size of intensity: ', np.size(intensity) , '\n')
# print('t:',t)
# print('intensity:', intensity)

# print(point_cloud)
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z, s=0.1)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 20)
plt.show()
'''

#transform to 1D array:
x = x[0]
y = y[0]
z = z[0]
t = t[0]
intensity = intensity[0]

#make them like the form [[x[0],y[0],z[0]], [x[1],y[1],z[1]], ... ]  :
point_cloud_list = []
for i in range(len(x)):
    point_cloud_list.append([x[i],y[i],z[i]])
    

# array --> vector
point_cloud_list_vector = np.array(point_cloud_list)
#print(point_cloud_list_vector)

pcd = o3d.geometry.PointCloud()
#pcd.points
pcd.points = o3d.utility.Vector3dVector(point_cloud_list_vector[:,:3])  #xyz

#pcd.colors 
# we visualize the intensity in 3D using pseudo color. Violet indicates low density and yellow indicates a high density.
intensity = np.asarray(intensity) #array-->vector
intensity_colors = plt.get_cmap('plasma')(
    (intensity - intensity.min()) / (intensity.max() - intensity.min()))
intensity_colors = intensity_colors[:, :3]
pcd.colors = o3d.utility.Vector3dVector(intensity_colors)

#pcd.normals
#which locally fits a plane per 3D point to derive the normal. However, the estimated normals might not be consistently oriented.
pcd.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))  # invalidate existing normals
pcd.estimate_normals()

#quick view
o3d.visualization.draw_geometries([pcd])

#o3d.visualization.draw_geometries([pcd], point_show_normal=True)



#Strategy1:BPA(ball-Pivoting Algorithm)
#first compute the necessary radius parameter based on the 
#average distances computed from all the distances between points:

distances = pcd.compute_nearest_neighbor_distance()
#print("distances\n",distances)#0.23, 0.28.....
avg_dist = np.mean(distances)#0.0165
#print('\n\navg_dist\n',avg_dist)
radius =[avg_dist, 2*avg_dist, 5*avg_dist, 10*avg_dist, 13*avg_dist,16*avg_dist,18*avg_dist,20*avg_dist,25*avg_dist,28*avg_dist,34*avg_dist,40*avg_dist]
#radius = 0.300000

#pcd.normals = o3d.geometry.TriangleMesh.compute_triangle_normals(pcd.points)
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility
.DoubleVector(radius))  #create mesh and store it  in bpa_mesh

#downsample the result to an acceptable number of triangles, exp.100k
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

#Additionally, if you think the mesh can present some weird artifacts, you can 
#run the following commands to ensure its consistency:
#dec_mesh.remove_degenerate_triangles()
#dec_mesh.remove_duplicated_triangles() #去掉重复三角
#dec_mesh.remove_duplicated_vertices()  #去掉重复顶点
#dec_mesh.remove_non_manifold_edges()
'''
#Startegy2:Possion'reconstruction
#computing the mesh
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=False)[0]

#cropping
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)
'''
#output
output_path = "mesh_out_put/00000/"
o3d.io.write_point_cloud(output_path+"pcd.ply", pcd)
o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", dec_mesh)
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
my_lods = lod_mesh_export(dec_mesh, [10000,3000, 1000,100], ".ply", output_path)
#my_lods = lod_mesh_export(p_mesh_crop, [100000,10000,3000, 1000,100], ".ply", output_path)

#visualize within python a specific LoD, let us say the LoD with 100 triangles, you can access and visualize it through the command
#o3d.visualization.draw_geometries([my_lods[1000]])
