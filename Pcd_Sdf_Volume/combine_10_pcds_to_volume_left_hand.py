# -*- coding: utf-8 -*-
"""
use the combined pcd to create volume by SDF, prepare for the next step-->marching cubes

"""
import timeit
import numpy as np
import open3d as o3d
import math
import os
import trimesh
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from skimage import measure
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

#input_dir = "Input_for_combination/"
input_dir = "Input_straitline/"
output_dir = "Output_combination/"
lidar_pos_filename = "lefthand_lidar_pos_rot.txt"
lidar_data  = np.loadtxt(input_dir + lidar_pos_filename, skiprows=1)

# lidar world position at frame 003475
''' (pitch-Y,yaw-Z,roll-X); settings of lidar-->fov-up:5°, fov-down:-15°, range:30m, channel:64 
#003475 Transform(Location(x=26.593929, y=-5.600266, z=2.498643), Rotation(pitch=0.008224, yaw=-179.992188, roll=-0.001038))#, 
'''
lidar_pos = np.vstack([
    lidar_data[:,1], 
    lidar_data[:,2],
    lidar_data[:,3]]).T
#print("lidar_pos.shape: ", lidar_pos.shape)  #(146,3)

# rot degrees
rotation = np.vstack([
    lidar_data[:,4],
    lidar_data[:,5],
    lidar_data[:,6]]).T
#print("rotation.shape:", rotation.shape)  #(146,3)

# sample 10 measurements and get the filenames list, lidar_pos/rot_list
frames = np.array(lidar_data[:,0])
#print(frames) #[6606. 6607. 6608. 6609. 6610. 6611. 6612. 6613. 6614. 6615. 6616. 6617......
m_times = 10    #combine 10 measurements , use 2 to quckly test

filenames = next(os.walk(input_dir))[2]
#print(filenames)
lidar_filenames  = []
lidar_pos_list = []
lidar_rot_list = []
specified_frame = 16322  #begin with this frame

# def get_pcd(begin_frame, sample_times, sample_interval):
#     '''return pcds list with frames'''
for i in range(m_times):
    for filename in filenames:
        if filename.startswith('0'+str(specified_frame)) and filename.endswith(".xyz"): #Attention!!! should change with the specified frame 0 or 00      
            lidar_filenames.append(os.path.join(filename))
            print(lidar_filenames) #['006617_point.xyz', '006627_point.xyz']
    for count, f in enumerate(frames):
        #print(count, f)
        if int(f) == int(specified_frame):
            lidar_pos_list.append(lidar_pos[count])
            lidar_rot_list.append(rotation[count])
    specified_frame += 1   # interval is 1 frames: 6617, 6618....
    print("time%d: load point filename and lidar pos/rot successful!" % (i))

lidar_pos_list = np.vstack(lidar_pos_list)
print("lidar_pos_list.shape: ", lidar_pos_list.shape )
lidar_rot_list = np.vstack(lidar_rot_list)
print("lidar_rot_list.shape: ", lidar_rot_list.shape )

# pcd and lidar position data for each frame in filenames list
pcd_pos_range_list = []     # to save 10 measurements pcd world position
lidar_position_list = []    # to save 10 measurements' lidar world positions
max_range = 30             # [meters]

for count, f_name in enumerate(lidar_filenames):
    data = np.loadtxt(input_dir + f_name, skiprows=1) 
    raw_pcd_pos = np.array(data[:,:3])  # pcd position in lidar coords
    print("%d: load pcd data successful!" % (count))
    #print("raw_pcd_pos shape: ", raw_pcd_pos.shape) 
    
    # pcd position from lidar coordination system to world coordination system position
    r = R.from_euler('yzx', lidar_rot_list[count], degrees=True) # lidar orientation in euler
    pcd_world_pos = r.apply(raw_pcd_pos) + lidar_pos_list[count] # world coords system
    print("%d: convert to pcd_world_pos.shape: "% (count),pcd_world_pos.shape) 

    # # save lidar position
    # lidar_position_list.append(lidar_pos_list[count])
    # print("%d: load lidar position!"% (count)) 
    
    #set range[m], only use the points within a max_range
    pcd_pos_range = []          # to save 1 frame measurements pcd world position with max_range
    for p in pcd_world_pos:
        if np.linalg.norm(p - lidar_pos_list[count]) <= max_range:
            pcd_pos_range.append(p)
    pcd_pos_range = np.vstack(pcd_pos_range)    # 每一个frame的变成一列
    #print("pcd_pos_range shape: ", pcd_pos_range.shape)
    pcd_pos_range_list.append(pcd_pos_range)    # 10 measurements pcd world position, 其中每一个frame是一列

pcd_pos_range_vstack = np.vstack(pcd_pos_range_list)  # base this to generate voxel's volume, which is big enough to save all voxels
print("pcd_pos_range_vstack.shape: ", pcd_pos_range_vstack.shape)


#set voxel size
set_voxel_size = 0.1
#print("voxel_size[m]:", set_voxel_size)

start = timeit.default_timer()
class Volume:
    '''create voxel volume from 3D point cloud data'''
    def __init__(self, voxel_size, pcd_pos_range_list, pcd_pos_range_vstack): #set_voxel_size, combined pcd_world_pos
        self.voxel_size = voxel_size
        self.pcd_pos_range_list = pcd_pos_range_list
        self.pcd_pos = pcd_pos_range_vstack       

        x_array = self.pcd_pos[:,0]
        y_array = self.pcd_pos[:,1]
        z_array = self.pcd_pos[:,2]
        #i_array = self.pcd_pos[:,3] #intensity
        vox_coords_origin = np.array([min(x_array),min(y_array), min(z_array)]) # the most left and most down postition 
        print("voxel coords origin, most left and down position: ", vox_coords_origin)
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
        

        #print("box volume in xyz[m]: %d x %d x %d" % (volume_Xaxis_length, volume_Yaxis_length, volume_Zaxis_length))
        #print("box volume: %d" % (volume_box))
        print("voxel count: %d" % (voxel_count))
        #print("Voxel size [m]: %f" % (self.voxel_size))
        

        #get voxel in Numpy array with meshgrid
        xv, yv, zv = np.meshgrid(range(voxel_Xaxis_count),range(voxel_Yaxis_count),range(voxel_Zaxis_count), indexing='ij')
        print("xv.shape:", xv.shape) #shape of the volume
        #print("yv.shape:", yv.shape)
        #print("zv.shape:", zv.shape)

        #get voxel cooridinate in this volume
        vox_coords = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)), axis=1)
        # may cause memory problem if choose a small voxel size like 0.02
        print("voxel_coords shape: ", vox_coords.shape) #(n,3) 
        print("vox_coords[0]: ",vox_coords[0]) #[0 0 0] -- [118, 95, 17] the most left and down voxel is [0 0 0]

        # voxel coordinates to world coordinates
        voxel_world_coords = vox_coords*voxel_size +vox_coords_origin
        print("voxel_world_coords[0]: ", voxel_world_coords[0])

        #voxel center coordinates to world coordinates
        voxel_center_world_coords = voxel_world_coords + 0.5*voxel_size
        print("voxel_center_world_coords[0]: ", voxel_center_world_coords[0])

        # define some parameters for the following sdf function, should rewrite in the future
        self.lidar_pos = lidar_pos
        self.voxel_center_world_coords = voxel_center_world_coords
        self.voxel_count = voxel_count
        self.vox_coords = vox_coords
        self.vox_world_coords = voxel_world_coords
        self.xv = xv
        
    '''
    def voxel_grid(self):
        # save the voxels within radius, for o3d drawing voxel grid
        voxel_coords = []
        for ind_v in indices_v:
            voxel_coords.append(self.voxel_center_world_coords[ind_v])
        voxel_coords = np.vstack(voxel_coords)
    '''
    # pcd and lidar_pos of each frame
    # sdf() 求每一个frame，每个voxel的sdf和w，以及可以求出D和W
    def sdf(self, pcd, lidar_pos): 
        # find the nearest tow points and the distance between them
        tree_point = KDTree(pcd)
        distances_pp, indices_pp = tree_point.query(pcd, k=1)
        min_radius = distances_pp/2
        print("pcd_pos.shape:",pcd.shape)
        print("min_radius.shape",np.squeeze(min_radius).shape)

        # find the voxels, which located inside the range r; for each point
        tree_voxel = KDTree(self.voxel_center_world_coords)
        # 1.if use the min_radius
        #indices_v = tree_voxel.query_radius(pcd, r = np.squeeze(2*min_radius))    # 得到的是在半径内的voxel_center的indices
        # 2.if directly set the radius as 5*voxel_size
        indices_v = tree_voxel.query_radius(pcd, r = 5*self.voxel_size)
        print("indices_v.shape", indices_v.shape)

        
        #find the nearest voxel center of each point, with KD-Trees for relative large pcd
        tree = KDTree(self.voxel_center_world_coords)
        distances, indices = tree.query(pcd, k=5) #k=1 to find the cloesest one  if k=5 -->(61341,5)
        #print("shape of indices should equal to number of pcds: ", indices.shape)

        # marchingcubes error if indices_v[n] array is empty, so when we use min_radius, need the following for loop
        #print(indices_v[0])
        for i in range(len(indices_v)):
            if indices_v[i].size == 0:
                indices_v[i] = indices[i]                                       # 得到的是对于半径内没有任何voxel_center的，给他们点周围的k个voxel_center的indices
        #print(indices_v[0])
        

        '''
        # save the voxels within radius, for o3d drawing voxel grid
        voxel_coords = []
        for ind_v in indices_v:
            voxel_coords.append(self.voxel_center_world_coords[ind_v])
        voxel_coords = np.vstack(voxel_coords)
        '''        
        
        #project point on line(lidar<-->nearest voxel center) and compute the sign distance
        #这里np.full 设置成100 和100.0 对结果影响还挺大，到底是影响到哪儿了呢？？？
        sign_dist = np.full((self.voxel_count,),100.0) #initial sd = 100.0
        w = np.full((self.voxel_count,),0.0) # initial weight of each voxel = 0.0
        print("sign_dist.shape",sign_dist.shape)

        # 这里的indices_v是一次扫描得到的每个点附近范围内的voxel的indices， 每个indices里有多个单一voxel的indice
        # 这里的pcd是一次扫描的点云，所以pcd[count]可以表示和indices_v对应的点
        # lidar_pos 是一次扫描的lidar位置
        for count, ind in enumerate(indices_v):
            #count: order of each pcd, ind: indices array of voxels' indices within min_radius or specific radius
            for ind_i in ind:
                nearest_voxel = self.voxel_center_world_coords[ind_i]
                vector_lidar_point = np.squeeze(pcd[count] - lidar_pos)
                vector_lidar_voxel = np.squeeze(nearest_voxel - lidar_pos)
                project_point = (np.squeeze(lidar_pos) + np.dot(vector_lidar_point,vector_lidar_voxel)/np.dot(
                                vector_lidar_voxel,vector_lidar_voxel)*vector_lidar_voxel)
                #print(project_point.shape) #(1,3)

                #calculate the sign distance of each voxel
                dist_p = np.linalg.norm(lidar_pos - project_point)
                dist_v = np.linalg.norm(lidar_pos - nearest_voxel)
                single_sign_dist = dist_v - dist_p
                sign_dist[ind_i] = single_sign_dist

                #calculate the weight of each voxel, w = 1 - (depth/max_range)
                dist_p_lidar = np.linalg.norm(pcd[count] - lidar_pos)
                w[ind_i] = 1.0 - (dist_p_lidar / max_range)
                

        # reshape to volume and marching cubes to get mesh of each scan
        sdf_volume = sign_dist.reshape(self.xv.shape) #assign these distances to volume
        print("sdf_volume.shape:", sdf_volume.shape)       
        return sign_dist, w, sdf_volume


# test class
V = Volume(set_voxel_size, pcd_pos_range_list, pcd_pos_range_vstack)

sdf_list = []
w_list = []
#initial big_D and big_W
big_W = np.full((V.voxel_count,),0.0)
big_D = np.full((V.voxel_count,),100.0)

for time in range(m_times):
    pcd = pcd_pos_range_list[time]
    lidar_pos = lidar_pos_list[time]

    sign_dist, w , sdf_volume= V.sdf(pcd, lidar_pos)
    #np.savetxt(output_dir+'w.txt', w)
    #np.savetxt(output_dir+'sign_dist.txt', sign_dist)
    # marching cubes to get mesh of each time
    verts, faces, normals, values = measure.marching_cubes(sdf_volume, level=0.15)
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    mesh.export(file_obj=output_dir+"sdf-mesh_range%d_voxsize%s_time%d.ply" % (max_range, set_voxel_size, time))

    # save each time's sdf and w into list
    sdf_list.append(sign_dist)
    w_list.append(w)

    # calculate W and D of each voxel at each time
    #big_W = w + big_W
    for i in range(len(big_D)):     
        # refresh big_D[i]
        if big_W[i] + w[i] == 0:    #no point in this voxel
            big_D[i] = 100
        else:
            big_D[i] = (big_D[i] * big_W[i] + sign_dist[i] * w[i]) / (big_W[i] + w[i])       
        # refresh big_W[i]
        big_W[i] = big_W[i] + w[i]
  
    # save point cloud file of each time 
    point_each_time = o3d.geometry.PointCloud()
    point_each_time.points = o3d.utility.Vector3dVector(pcd)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_dir+"pcd_maxrange%d_time%d.ply" % (max_range, time), point_each_time)



# save combined point cloud file of total time
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(pcd_pos_range_vstack)
o3d.io.write_point_cloud(output_dir+"combined_%d_pcds.ply" % (m_times), combined_pcd)

# now we have big_D array, reshape it as volume
big_D_volume = big_D.reshape(V.xv.shape) #assign these distances to volume
#np.savetxt(output_dir+'big_D.txt', big_D)
print(big_D_volume.shape)
print(big_D_volume.size)

# mesh
verts_D, faces_D, normals_D, values_D = measure.marching_cubes(big_D_volume, level=0.1)
mesh_D = trimesh.Trimesh(vertices=verts_D,faces=faces_D)
mesh_D.export(file_obj=output_dir+"combined_%d_mesh.ply" % (m_times))    
    
# stop time
stop = timeit.default_timer()
print("Time:", stop-start)








# marching cubes, parameters as follow:
#(volume, level=None, *, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, method='lewiner', mask=None)

#save verts
#np.savetxt(output_dir + 'verts.xyz', verts, delimiter=' ')

# # show with mlab
# mlab.triangular_mesh([vert[0] for vert in verts],
#                         [vert[1] for vert in verts],
#                         [vert[2] for vert in verts],
#                         faces)
# mlab.show()

# show in pcd and voxel grid with open3d
'''
#view voxel grid within radius
p_vox = o3d.geometry.PointCloud()
p_vox.points = o3d.utility.Vector3dVector(voxel_coords)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(p_vox, voxel_size=set_voxel_size)
#o3d.visualization.draw_geometries([voxel_grid])
o3d.io.write_voxel_grid(output_dir+"lefthand_voxel_grid_range%d_voxsize%s.ply" %(max_range,set_voxel_size), voxel_grid)
'''






