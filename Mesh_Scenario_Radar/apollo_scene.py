import os
import sys
python_file_directory = os.path.dirname(os.path.abspath(__file__)) # output the dir of this file
upper_directory = python_file_directory + "/../"
sys.path.append(upper_directory)

from radar_ray_python.Persistence import save_radar_measurement_as_binary
from radar_ray_python.Renderer import RenderMode, Renderer
from radar_ray_python.RxAntenna import RxAntenna
from radar_ray_python.TxAntenna import TxAntenna
from radar_ray_python.Material import MaterialDielectric, MaterialLambertian, MaterialMetal, MaterialMixed
import radar_ray_python as raray
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import time
from PIL import Image
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

from radar_ray_python.data.TraceData import *
from radar_ray_python.data.SignalData import *
from radar_ray_python.Persistence import load_mesh_normal_from_obj



def plot_rms_and_std(error_array):
   x = np.array(range(error_array[:,0].shape[0]))
   y = error_array[:,0]
   e = error_array[:,1]
   plt.errorbar(x, y, e, linestyle='None', marker='^')

   plt.savefig("error_plot.png")

def plot_sphere_radius_performance():
   """
   method for paper plot
   """
   mpl.rc('font',family='Times New Roman')
   # gold-standard: 284sek # arround 11000 rays
   sphere_sizes = [0.5, 1.0, 2.0, 4.0]
   elapsed_sphere_times = [12.24, 4.0, 1.17, 0.43]
   elapsed_sphere_times_fast = [5.44, 1.96, 0.56, 0.28]

   elapsed_cone_times = [4.0, 2.0, 0.8, 0.71]
   elapsed_cone_times_fast = [2.23, 1.14, 0.52, 0.43]

   plt.plot(sphere_sizes, elapsed_sphere_times, marker='o', label="vary sphere size")
   plt.plot(sphere_sizes, elapsed_sphere_times_fast, marker='x', label="vary sphere size optimized")
   plt.plot(sphere_sizes, elapsed_cone_times, marker='o', label="vary cone angle")
   plt.plot(sphere_sizes, elapsed_cone_times_fast, marker='x', label="vary cone angle optimized")
   ax = plt.gca()
   ax.set_xticks(sphere_sizes)
   ax.set_xticklabels(['0.5/2°', '1.0/4.0°', '2.0/8°' ,'4.0/16°'])

   plt.legend(prop={"size":14})
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.grid()
   plt.xlabel("Sphere radius (m)/Cone Angle (deg.)", size=14)
   plt.ylabel("Time in s", size=14)
   plt.savefig("SphereSizePerformance.pdf", bbox_inches='tight')

def plot_antenna_configuration(radar_signal_data):

   mpl.rc('font',family='Times New Roman')
   tx_positions_2d = radar_signal_data.tx_positions[:, :]
   rx_positions_2d = radar_signal_data.rx_positions[:, :]

   tx_positions_2d -= tx_positions_2d[0]
   rx_positions_2d -= rx_positions_2d[0]
   csfont = {'fontname':'Times New Roman'}

   plt.scatter(rx_positions_2d[:, 0], rx_positions_2d[:, 1], marker='o', label="rx antenna positions")
   ax = plt.gca()
   ax.set_yticklabels([])
   ax.set_yticks([])
   ax.set_xlabel("x in m", size=14, **csfont)
   plt.xticks(**csfont)

   ax.scatter(tx_positions_2d[:, 0], tx_positions_2d[:, 1], marker='x', label="tx antenna positions")

   virtual_positions = []
   for rx_pos in rx_positions_2d:
      for tx_pos in tx_positions_2d:
         virtual_pos = (rx_pos + tx_pos)
         virtual_positions.append(virtual_pos)

   virtual_positions = np.asarray(virtual_positions)
   
   ax.scatter(virtual_positions[:, 0], virtual_positions[:, 1]-0.1, marker='o', color='blue', label="virtual antenna positions")
   ax.legend()
   ax.set_aspect(0.33)
   plt.savefig("antenna_positions.eps", bbox_inches='tight')


def get_max_range(radar_signal_data):
   c=3e8
   chirp_duration = radar_signal_data.chirp_duration
   bandwidth = radar_signal_data.bandwidth
   sample_frequency = radar_signal_data.time_vector.shape[0]/chirp_duration
   r_max = (c*chirp_duration*sample_frequency)/(4*bandwidth)*2 # complex signal multiply by 2

   return r_max

def compute_azimuth_label_sine_space(number_sample_points, angular_dim, start_sin_index=0, stop_sin_index=None):
   
   if not stop_sin_index:
      stop_sin_index = angular_dim

   start_sin_value = -2.0/angular_dim*start_sin_index + 1
   end_sin_value = -2.0/angular_dim*stop_sin_index + 1
   #sin_space_labels = np.linspace(1, -1, number_sample_points)

   sin_space_labels = np.linspace(start_sin_value, end_sin_value, number_sample_points)
   angular_labels = np.round(np.rad2deg(np.arcsin(sin_space_labels))).astype(np.int32)
   angular_positions = np.linspace(0, angular_dim-1, number_sample_points)

   #angular_labels = np.round(np.linspace(90, -90, number_sample_points))
   #angular_positions = (np.sin(np.deg2rad(angular_labels)) + 1.0)*0.5 * (angular_dim-1)
   return angular_positions, angular_labels

def compute_range_label(radar_signal_data, number_sample_points, range_dim): #radar_data, 9, 512
   r_max = get_max_range(radar_signal_data)
   print("r_max: " + str(r_max))
   range_labels = np.round(np.linspace(0, r_max, number_sample_points)).astype(np.int32)
   range_positions = np.linspace(0, range_dim, number_sample_points)
   return range_positions, range_labels, r_max

def load_campus_scene(renderer, material_dir, obj_filename):
   mesh_list, obj_mat_list = load_mesh_normal_from_obj(obj_filename, material_dir)

   for i, obj_mat in enumerate(obj_mat_list):
      mesh = mesh_list[i]
      if "Glass" in obj_mat.name:
         mesh_mat = MaterialDielectric(1.5)
      elif "Metal" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)
      elif "ContainerMat" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.15) # 0.2
      elif "LanternMat" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.15) # 0.1
      elif "Pavement" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.7)
      elif "Lawn" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.7)
      elif "Concrete" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.7)
      elif "WindowFrame" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.2)
      elif "t-shirt" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.5)
      elif "pants" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.5)
      elif "Skin" in obj_mat.name:
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.5)
      else:
         #mesh_mat = MaterialMetal(obj_mat.diffuse, 0.1)
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.6)

      mesh.set_material(mesh_mat)
      renderer.add_geometry_object(mesh)

def set_antennas_high_resolution(renderer, offset_pos, rx_radius=1.0):
   number_tx_horizontal = 64
   number_tx_vertical = 1
   number_rx_horizontal = 64
   number_rx_vertical = 1
   antenna_look_dir = np.array([0.0, -1.0, 0.0])
   c=3e8
   f=77e9
   w = c/f
   antenna_distance =2e-3 #w*0.5
   rx_dist_horizontal = number_tx_horizontal*antenna_distance
   rx_dist_vertical = number_tx_vertical*antenna_distance

   tx_dist_horizontal = antenna_distance
   tx_dist_vertical = antenna_distance

   rx_antennas = []
   for rx_index_v in range(number_rx_vertical):
      for rx_index_h in range(number_rx_horizontal):
         rx_pos = np.array([rx_index_h*rx_dist_horizontal, 0.0, rx_index_v*rx_dist_vertical])
         rx_pos = rx_pos + offset_pos
         rx_antenna = RxAntenna(rx_pos, rx_radius)
         rx_antenna.enable_cone_mode(np.deg2rad(6.0))
         rx_antennas.append(rx_antenna)
         renderer.add_rx_antenna(rx_antenna)

   for tx_index_v in range(number_tx_vertical):
      for tx_index_h in range(number_tx_horizontal):
         tx_pos = np.array([tx_index_h*tx_dist_horizontal, 0.0, tx_index_v*tx_dist_vertical])
         tx_pos = tx_pos + offset_pos

         tx_antenna = TxAntenna(tx_pos)
         look_at = tx_pos + antenna_look_dir
         tx_antenna.set_look_at(look_at)
         tx_antenna.set_up(np.array([0.0, 0.0, 1.0]))
         tx_antenna.set_elevation(np.deg2rad(6.0))
         tx_antenna.set_azimuth(np.deg2rad(80.0))
         renderer.add_tx_antenna(tx_antenna)

   return rx_antennas

def set_antennas_quad_digimmic_3_16(renderer, offset_pos, rx_radius=1.0, cone_angle_deg=None):
   #antenna_look_dir = np.array([0.0, -1.0, 0.0])  #campus, !in blender(0,0,-1)
   #antenna_look_dir = np.array([0.0, -1.0, 0.0])  #carla
   #antenna_look_dir = np.array([0.0, 0.0, 1.0])  #kitti
   antenna_look_dir = np.array([0.0, -1.0, 0.0])  #apolo
   #antenna_look_dir = np.array([0.0, -1.0, 0.0])  #LHFT,  !in belnder(-1,0,0)

     
   tx_antenna_0 = TxAntenna(np.array([0.0e-3, 0.0, 0.0] + offset_pos))
   tx_antenna_1 = TxAntenna(np.array([2.0e-3, 0.0, 0.0] + offset_pos))
   tx_antenna_2 = TxAntenna(np.array([4.0e-3, 0.0, 0.0] + offset_pos))
   tx_antennas = [tx_antenna_0, tx_antenna_1, tx_antenna_2]

   for tx_antenna in tx_antennas:
      tx_pos = tx_antenna.get_position()
      look_at = tx_pos + antenna_look_dir
      tx_antenna.set_look_at(look_at)
      tx_antenna.set_up(np.array([0.0, 0.0, 1.0]))
      tx_antenna.set_elevation(np.deg2rad(6.0))
      tx_antenna.set_azimuth(np.deg2rad(80.0))
      renderer.add_tx_antenna(tx_antenna)

   rx_antennas = []
   rx_antennas.append(RxAntenna(np.array([0e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([6e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([12e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([18e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([24e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([30e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([36e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([42e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([48e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([54e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([60e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([66e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([72e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([78e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([84e-3, 0.0, 0.0]) + offset_pos, rx_radius))
   rx_antennas.append(RxAntenna(np.array([90e-3, 0.0, 0.0]) + offset_pos, rx_radius))

   for rx_antenna in rx_antennas:

      if cone_angle_deg is not None:
         rx_antenna.enable_cone_mode(np.deg2rad(cone_angle_deg))
      renderer.add_rx_antenna(rx_antenna)

   return rx_antennas

#polar koordinatensystem --> Kartesisches Koordinatensystem
def pol2cart(rho, angle):
   x = rho * np.cos(angle)
   y = rho * np.sin(angle)
   return(x, y)

def cart2pol(x, y):
   rho = np.sqrt(x**2 + y**2)
   phi = np.arctan2(y, x)
   return(rho, phi)

def create_radar_image(radar_sim_data):

   # python: index 0: tx-antenna, index 1: rx-antenna, index 2: chirp/doppler, index 3: time signal/range
   # matlab: index 1: tx-antenna, index 2: rx-antenna, index 3: chirp/doppler, index 4: time signal/range

   if_signal = radar_sim_data.signals
   number_tx = if_signal.shape[0] # 8
   number_rx = if_signal.shape[1] # 8
   number_chirps = if_signal.shape[2] # 64
   number_samples = if_signal.shape[3] # 340
   number_virt_antennas = number_tx*number_rx
   virt_signal = np.zeros((number_tx*number_rx, number_chirps, number_samples), dtype=np.complex128) #(64, 64, 340)

   #virt_signal: index 0: tx*rx, index 1: chirp/doppler, index 2: time signal/range

   for tx_index in range(number_tx):
      for rx_index in range(number_rx):
         virt_index = tx_index*number_rx + rx_index
         virt_signal[virt_index] = if_signal[tx_index, rx_index]


   # zero padding and windowing
   zero_padded_virt_signal = np.zeros((512, 512, 512), dtype=np.complex128) #128, 128, 512, 0.+0.j 零填充，在卷积的时候，为了保证图像大小不发生变化。 零填充填几层是自动计算的。
   '''###Frage?###'''
   '''https://numpy.org/doc/stable/reference/generated/numpy.hanning.html?highlight=hanning#numpy.hanning'''
   
   #对数字信号进行快速傅里叶变换，可得到数字信号的分析频谱。分析频谱是实际频谱的近似。傅里叶变换是对延拓后的周期离散信号进行频谱分析。如果采样不合适，某一频率的信号能量会扩散到相邻频率点上，出现频谱泄漏现象。
   #为了减少频谱泄漏，通常在采样后对信号加窗。傅里叶分析的频率分辨率主要是受窗函数的主瓣宽度影响，而泄漏的程度则依赖于主瓣和旁瓣的相对幅值大小。矩形窗有最小的主瓣宽度，但是在这些最常见的窗中，矩形窗的旁瓣最大。
   # 因此，矩形窗的频率分辨率最高，而频谱泄漏则最大。不同的窗函数就是在频率分辨率和频谱泄漏中作一个折中选择。
   hanning_single_angle = np.hanning(number_virt_antennas+2)[1:-1, np.newaxis] #(64,1) 
   hanning_single_doppler = np.hanning(number_chirps+2)[1:-1, np.newaxis]  
   hanning_single_range = np.hanning(number_samples+2)[1:-1, np.newaxis]
   hanning_window = hanning_single_angle@hanning_single_doppler.T
   hanning_window = hanning_window[..., np.newaxis]
   hanning_window = hanning_window@hanning_single_range.T

   zero_padded_virt_signal[:virt_signal.shape[0], :virt_signal.shape[1], :virt_signal.shape[2]] = hanning_window*virt_signal 
   #print(zero_padded_virt_signal.shape) -> (128, 128, 512)

   # create 3D-fft angle, doppler, range
   reco_3d = np.fft.fftn(zero_padded_virt_signal)
   reco_3d = np.abs(reco_3d)  
   reco_3d = np.fft.fftshift(reco_3d, axes=(0,1))
   #reco_3d = np.flip(reco_3d, axis=0)
   #print(reco_3d.shape) #--> (128, 128, 512)
   #print(reco_3d)
   


   # create slice for angle and range
   reco_angle_range = np.max(reco_3d, axis=1) #(128, 512), 2D array
   #print(reco_angle_range)
   #reco_angle_range = np.sum(np.abs(reco_3d), axis=1)
   max_value = np.max(reco_angle_range)
   #print(max_value)

   #change x,y axes' size
   # x: 128--0 -> -90--90;  y:0--512 -> 0--r_max;  Doppler:128--0 -> v_max--(-v_max)
   def changex(temp, position):
      return int(temp/10)

   def changey(temp, position):
      return int(((256-temp)/512)*180)

   def changev(temp, position):
      return round(((256-temp)/256)*1.623376, 2)



   # fs = number_samples/chirp_duration
   # c=3e8
   # wavelength = c/carrier_frequency
   # mu = bandwidth/chirp_duration
   # create axes
   # r_max =  (fs*c)/(2*mu)
   # v_max = wavelength/(4*chirp_duration) 

   r_max = 51
   v_max = 1.623376

   plt.figure("reco_angle_range")
   ax = plt.gca()
   plt.imshow(reco_angle_range, aspect="auto", vmax=max_value/2.0)
   ax.xaxis.set_major_formatter(FuncFormatter(changex))
   ax.yaxis.set_major_formatter(FuncFormatter(changey))
   plt.xlabel("range/[m]")
   plt.ylabel("angle/[°]")
   #plt.xlim(0,512)
   #plt.ylim(128,0)
   plt.savefig("image2/reco_angle_range{}.png".format(0), bbox_inches="tight")
   #plt.savefig("testimage/reco_angle_range{}.png".format(count), bbox_inches="tight")
   plt.close()

   # create slice for Doppler and range
   reco_doppler_range = np.max(reco_3d, axis=0)
   #print(reco_doppler_range.shape) -> (128, 512)
   max_value = np.max(reco_doppler_range)

   plt.figure("reco_doppler_range")
   ax = plt.gca()
   plt.imshow(reco_doppler_range, aspect="auto", vmax=max_value/2.0)
   ax.xaxis.set_major_formatter(FuncFormatter(changex))
   ax.yaxis.set_major_formatter(FuncFormatter(changev))
   plt.xlabel("range/[m]")
   plt.ylabel("Doppler/[m/s]")
   # plt.xlabel("y_position")
   # plt.ylabel("x_position")
   plt.savefig("image2/reco_doppler_range{}.png".format(0), bbox_inches="tight")
   #plt.savefig("testimage/reco_doppler_range{}.png".format(count), bbox_inches="tight")
   plt.close()

   # generate detections
   max_value_of_reco = np.max(reco_3d)
   #print(max_value_of_reco)
   threshold = max_value_of_reco*0.1  # or any suitable value
   indices = np.argwhere(reco_3d > threshold) #argwhere return the indices of element， 索引值
   #print(reco_3d.shape) -> (128, 128, 512)

   
   # #save in txt
   # indices_angle_doppler_range = indices
   # filename = 'angle_range.txt'
   # with open(filename, 'a') as f_obj:
   #     f_obj.writelines('radar frame: ' + count + '\n')
   #     np.savetxt(f_obj, indices_angle_doppler_range, fmt='%f', delimiter=',')


   #scatter plot
   #angle range need to trans to cartesian coordinate with x and y
   indices_angle_range = indices[:, (0,2)]
   #print(indices_angle_range[:, 1])   #range array这个值就是range
   #print(indices_angle_range[:, 0])   #sin-angle array 这个值是sin(angle), -90--90 -->-1--1 according to 0-->128
   sintheta = (256-indices_angle_range[:, 0])/256
   r = indices_angle_range[:, 1]/10 # range in meters
   angle = np.arcsin(sintheta) #弧度 radians
   x, y = pol2cart(r, angle)
   plt.figure("scatter_detections")
   plt.scatter(x, y)
   plt.xlim(0, 51)
   plt.ylim(-10, 10)
   
   plt.xlabel("x")
   plt.ylabel("y")

   plt.savefig("image2/scatter_plot{}.png".format(0), bbox_inches='tight')
   #plt.savefig("testimage/scatter_plot{}.png".format(count), bbox_inches='tight')
   plt.close()



   
def simulate(
   resolution=(800, 600), 
   oversampling_factor=16,
   rx_radius=0.5,
   cone_angle_deg=None,
   number_sequences=7,
   enable_color_bar=True,
   enable_range_axis=True,
   font_size=20,
   filename="test.png",
   compare_data=None):

   renderer = raray.Renderer()
   render_mode = RenderMode.RENDER_MODE_RADAR_FAST
   #render_mode = RenderMode.RENDER_MODE_GRAPHICS
   renderer.set_render_mode(render_mode)
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   #content_directory = os.path.join(script_directory, "../example-files/campus-scene/")
   content_directory = os.path.join(script_directory, "../example-files/lidar_reconstructed_scene/")
   #obj_filename = os.path.join(content_directory, "test_scene_campus.obj")   #Campus
   #obj_filename = os.path.join(content_directory, "untitled.obj") #CARLA
   #obj_filename = os.path.join(content_directory, "kitti.obj") #KITTI
   obj_filename = os.path.join(content_directory, "apolo.obj") #apolo
   #obj_filename = os.path.join(content_directory, "lhft.obj") #LHFT
   #print("start loading campus scene")
   load_campus_scene(renderer, content_directory, obj_filename)
   #print("finished loading campus scene")

   # set tx and rx antennas

   #!!! Camera and radar will block each other at the same position!!!
   #campus
   # antenna_offset = np.array([-8.8, 32.0, 0.7])   #campus
   # renderer.set_camera([-8.8, 32.0, 1.5], [-10.0, 0.0, 0.0], [0.0, 0.0, 1.0])   #campus

   #CARLA
   #antenna_offset = np.array([-140.0, 13.0, 15.0])    #CARLA y is up
   #renderer.set_camera([-140.0, 13.0, 15.0], [-10.0, 15.0, 15.0], [0.0, 1.0, 0.0]) #CARLA y is up, vector, from, to, up

   #KITTI
   # antenna_offset = np.array([133.0, 5.0, 2.2])    #KITTI
   # renderer.set_camera([133.0, 5.0, 2.2], [133.0, 4.9, 2.6], [0.0, 1.0, 0.0]) 

   #apolo
   antenna_offset = np.array([-9.2, 23, 1.5])    #apolo
   renderer.set_camera([-9.2, 23, 2], [-10, 0.0, 0.0], [0.0, 0.0, 1.0]) 

   #LHFT
   # antenna_offset = np.array([-4.6, 19.0, 1.4])    #LHFT
   # renderer.set_camera([-4.6, 19.0, 2], [-24.0, 0.0, 0.0], [0.0, 0.0, 1.0]) 


   #Test
   # antenna_offset = np.array([-4.6, 18.0, 2])    #LHFT
   # renderer.set_camera([2.406125068664551, 0.21050842106342316, -2.026690721511841], [-2.0360240936279297, 0.37147045135498047, -2.378622055053711], 
   # [-0.07996766269207001, -3.123927593231201, -0.4194153845310211]) 


   rx_antennas = set_antennas_quad_digimmic_3_16(renderer, antenna_offset, rx_radius=rx_radius, cone_angle_deg=cone_angle_deg)
   
   
   
  

   if renderer.get_render_mode() == RenderMode.RENDER_MODE_GRAPHICS:
      renderer.set_image_size(1200, 600) #1200,600
      renderer.set_oversampling_factor(200)
      renderer.initialize()
      renderer.render()
      image = renderer.get_image()
      image_array = Image.fromarray(image)
      #image_array.save("campus_scene_image.png")
      #image_array.save("CARLA_scene.png")
      #image_array.save("KITTI_scene.png")
      image_array.save("apolo_scene.png")
      #image_array.save("LHFT_scene.png")
      print(image.shape)
      #plt.imshow(image)
      #plt.savefig("campus_scene_image.png")
   else:
      renderer.set_image_size(resolution[0], resolution[1]) #4m-800x650 8deg-800x650 16deg-720x600
      renderer.set_minimum_trace_length(16.0)
      renderer.set_oversampling_factor(oversampling_factor) # cone-mode: 2deg-9 4deg-5 8deg-2 16deg-1  no-cone-mode: 0.5m-16 1m-12 2.0m-8 4m-2
      renderer.set_trace_direction_computation(False)
      renderer.set_number_sequences(number_sequences) #cone-mode:2deg-1 4deg-1 8deg-1 16deg-1 no-cone-mode:0.1m-115 0.5m-5 1m-2 2.0m-1 4m-1

      time_start = time.time()
      renderer.render()
      time_end = time.time()
      print(f"the simulation took {time_end - time_start}")
      
      binary_filename = "traces_frame_{:03d}_chirp_{:03d}.bin".format(0, 0)   #这里没有人和车在场景里动，只输出一张就可以了
      save_radar_measurement_as_binary(renderer, binary_filename)

      trace_lengths = rx_antennas[0].get_trace_lengths()
      print(f"number of traces {len(trace_lengths[0])}")

      carrier_frequency = 77e9
      bandwidth = 1e9
      chirp_duration = 7.68e-5
      number_samples = 340
      trace_data = load_trace_data_from_binary_file(binary_filename)
      
      signal_data = create_fmcw_beat_signal_from_trace_data_cuda([trace_data], carrier_frequency, bandwidth, chirp_duration, number_samples)      
      print("finished signal generation")


      #plot fcuntion
      #create_radar_image(signal_data)

      #range-angle plot
      signals_array = signal_data.signals    #here should be if-signal
      signals_array = np.reshape(signals_array, (-1, signals_array.shape[2], signals_array.shape[3]), order='F')
      signals_array = signals_array[:, 0, :] # ignore doppler
      window_angle = np.hanning(signals_array.shape[0])[:, np.newaxis]  #angle
      window_range = np.hanning(signals_array.shape[1])[:, np.newaxis]  #range

      filter_window = window_angle@window_range.T
      windowed_signal = signals_array*filter_window

      range_dim = 512
      angular_dim = 512
      fft_range_angle = np.fft.fft2(windowed_signal, s=(range_dim, angular_dim))
      fft_range_angle = np.fft.fftshift(fft_range_angle, axes=(0))
      fft_range_angle_abs = np.abs(fft_range_angle)
      print("shape: ", fft_range_angle_abs.shape) #(512,512)
      #print("angle_range: ", fft_range_angle)
      #print("angle_range_abs: ", fft_range_angle_abs)

      norm_factor = (np.max(fft_range_angle_abs) - np.min(fft_range_angle_abs))

      fft_range_angle_abs_norm = fft_range_angle_abs / norm_factor
      fft_range_angle_abs_norm_log = 20*np.log10(fft_range_angle_abs_norm)
      print(fft_range_angle_abs_norm_log.shape) #(512,512)

      #crop to be equal with other measured image.
      offset = 7
      fft_range_angle_abs_norm_log = fft_range_angle_abs_norm_log[offset:-offset,:]
      angular_dim = fft_range_angle_abs_norm_log.shape[0]
      start_sin_index = offset
      stop_sin_index = angular_dim - offset

      #np.save("gold_standard.npy", fft_range_angle_abs)
      range_positions, range_labels, r_max = compute_range_label(signal_data, 9, range_dim)
      angular_positions, angular_labels = compute_azimuth_label_sine_space(9, angular_dim, start_sin_index, stop_sin_index)

      # flip image so that it looks like in the camera image
      fft_range_angle_abs_norm_log= np.flip(fft_range_angle_abs_norm_log.T, axis=0) #reverse the order of elements in an array
      range_labels = np.flip(range_labels)

      plt.imshow(fft_range_angle_abs_norm_log, vmin=-60, aspect=1)
      #plt.imshow(fft_range_angle_abs) #和 fft_range_angle_abs_norm_log 的结果方向不同，且不亮

      csfont = {'fontname':'Times New Roman'}
      
      ax = plt.gca()
      ax.set_yticks(range_positions)
      ax.set_yticklabels(range_labels, **csfont)
      #if enable_range_axis:
      ax.set_ylabel("Range in m", size=font_size, **csfont)

      ax.set_xticks(angular_positions)
      ax.set_xticklabels(angular_labels, **csfont)
      ax.tick_params(axis='both', which='major', labelsize=16)
      ax.set_xlabel("Angle in deg.", size=font_size, **csfont)

      cbar = plt.colorbar()
      cbar.ax.tick_params(labelsize=14)
      if enable_color_bar:
         cbar.set_label(label='Magnitude in dB', **csfont, size=font_size)
      
      # plt.xlim()
      # plt.ylim()
      plt.savefig(filename, bbox_inches='tight')
      plt.close()
      
      
      #scatter plot
      max_value_of_reco = np.max(fft_range_angle_abs)
      print("max value: ", max_value_of_reco)
      threshold = max_value_of_reco*0.1  # or any suitable value
      indices_angle_range = np.argwhere(fft_range_angle_abs > threshold) #argwhere return the indices of element， 索引值.  # indices of range and angle
      #indices_angle_range = np.argwhere(fft_range_angle_abs_norm_log)
      print(indices_angle_range.shape) #(262144, 2), 512*512 = 262144, add threshold -> (2820,2)
      #np.savetxt('indices_angle_rang.txt', indices_angle_range)

      #(512,512)
      sinangle = (0.5*range_dim-indices_angle_range[:, 0])/(0.5*range_dim)
      r = indices_angle_range[:, 1]/(range_dim/r_max) # range in meters
      angle = np.arcsin(sinangle) #弧度 radians
      x, y = pol2cart(r, angle)

      plt.figure("scatter_detections")
      plt.scatter(y, x, s=0.05)
      plt.ylim(0, 55)
      plt.yticks(np.arange(0,55,step=6))
      plt.xlim(-15, 15)
      plt.xticks(np.arange(-15,15,step=3))

      csfont = {'fontname':'Times New Roman'}
      plt.xlabel("x in m", size=font_size, **csfont)
      plt.ylabel("y in m", size=font_size, **csfont)

      plt.savefig("scatter_plot{}.png".format(0), bbox_inches='tight')
      #plt.savefig("testimage/scatter_plot{}.png".format(count), bbox_inches='tight')
      #plt.show()
      plt.close()
      
      # calculate rms and std-dev
      if compare_data is not None:
         diff = np.abs(fft_range_angle_abs_norm - compare_data)
         diff_mean = np.mean(diff)
         diff_std = np.std(diff)

         diff_mean_db = 20*np.log10(diff_mean)
         diff_std_db = 20*np.log10(diff_std)

         print(f"mean: {diff_mean}, std: {diff_std}")
         print(f"mean-db: {diff_mean_db}, std: {diff_std_db}")

         return fft_range_angle_abs_norm, np.array([diff_mean_db, diff_std_db])
      else:
         return fft_range_angle_abs_norm, np.array([0, 0])
      




def main():
   # normal mode
   # 0.1m
   # original from paper
   #comparison_image = simulate((800, 600), oversampling_factor=18, rx_radius=0.10, \
   #   number_sequences=120, enable_color_bar=True, enable_range_axis=False, filename="fft_image_campus_10cm_2.png")

   # adjusted for smaller gpus
   # comparison_image = simulate((800, 600), oversampling_factor=4, rx_radius=0.20, \
   #    number_sequences=130, enable_color_bar=True, enable_range_axis=False, filename="fft_image_campus_10cm_2.png")

   # adjusted for smaller gpus, CARLA_scene
   comparison_image = simulate((800, 600), oversampling_factor=4, rx_radius=0.20, \
      number_sequences=130, enable_range_axis=False, filename="apollo_scene_radar.png")
   # comparison_image = simulate((400, 300), oversampling_factor=4, rx_radius=0.20, \
   #    number_sequences=130, enable_range_axis=False, filename="apolo_scene_radar.png")
   # adjusted for smaller gpus, LHFT_scene
   # comparison_image = simulate((800, 600), oversampling_factor=4, rx_radius=0.20, \
   #    number_sequences=130, enable_range_axis=False, filename="LHFT_scene_radar.png")

   np.save("comparison_image.npy", comparison_image)
   #comparison_data = np.load("comparison_image.npy")

   error_list = []
   # vary sphere size
   # 0.5m
   #_, mean_std  = simulate((800, 600), oversampling_factor=18, rx_radius=0.5, \
   #    number_sequences=5, enable_color_bar=True, enable_range_axis=False, 
   #    filename="fft_image_campus_50cm_3_fast.png", compare_data=comparison_data)
   #error_list.append(mean_std)
#
   ## 1.0
   #_, mean_std = simulate((800, 600), oversampling_factor=13, rx_radius=1, \
   #   number_sequences=2, enable_color_bar=False, enable_range_axis=False, 
   #   filename="fft_image_campus_100cm_3.pdf", compare_data=comparison_data)
   #error_list.append(mean_std)
#
   ## 2.0
   #_, mean_std = simulate((800, 600), oversampling_factor=4, rx_radius=2, \
   #   number_sequences=2, enable_color_bar=False, enable_range_axis=False,
   #   filename="fft_image_campus_200cm_3.pdf", compare_data=comparison_data)
   #error_list.append(mean_std)
#
   ## 4.0
   #_, mean_std = simulate((800, 800), oversampling_factor=1, rx_radius=4, \
   #  number_sequences=2, enable_color_bar=True, enable_range_axis=False, 
   #  filename="fft_image_campus_400cm_3.pdf", compare_data=comparison_data)
   #error_list.append(mean_std)
#
   #error_array = np.asarray(error_list)
   #plot_rms_and_std(error_array)

   #oversampoing cone-mode: 2deg-9 4deg-5 8deg-2 16deg-1   # vary cone angle, sphere size=0.5
   # #cone-mode:2deg-1 4deg-1 8deg-1 16deg-1 
   # 2
   #_, mean_std = simulate((800, 600), oversampling_factor=18, rx_radius=0.5, cone_angle_deg=2, \
   #   number_sequences=1, enable_color_bar=False, enable_range_axis=True, 
   #   filename="fft_image_campus_50cm_3_cone_2.pdf", compare_data=comparison_data)

   # 4
   #_, mean_std = simulate((800, 600), oversampling_factor=9, rx_radius=0.5, cone_angle_deg=4, \
   #   number_sequences=1, enable_color_bar=False, enable_range_axis=False, 
   #   filename="fft_image_campus_50cm_3_cone_4.pdf", compare_data=comparison_data)
   # 8
   #_, mean_std = simulate((800, 600), oversampling_factor=3, rx_radius=0.5, cone_angle_deg=8, \
   #   number_sequences=1, enable_color_bar=False, enable_range_axis=False, 
   #   filename="fft_image_campus_50cm_3_cone_8.pdf", compare_data=comparison_data)   
   # 16
   #_, mean_std = simulate((800, 800), oversampling_factor=2, rx_radius=0.5, cone_angle_deg=16, \
   #  number_sequences=1, enable_color_bar=True, enable_range_axis=False, 
   #  filename="fft_image_campus_50cm_3_cone_16.pdf", compare_data=comparison_data)

   #error_array = np.asarray(error_list)
   #plot_rms_and_std(error_array)

if __name__ == "__main__":
   main()