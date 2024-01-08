#!/usr/bin/env python

# author: yao
# time: 19.01.2022
# lidar data collect use Carla
# output is with .xyz form, include x y z intensity


import glob
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import constants as cst
matplotlib.use('Agg')
from math import *

from queue import Queue
from queue import Empty

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import cv2
import carla


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480



def process_camera(image, sensor_queue, camera): 
    sensor_queue.put((image.frame, camera))
    image.save_to_disk('_out/image/%06d.png' % image.frame)




            # ^ z                       . z
            # |                        /
            # |              to:      +-------> x
            # | . x                   |
            # |/                      |
            # +-------> y             v y



def process_lidar_data(lidar_data, vehicle,sensor_queue, lidar):
    sensor_queue.put((lidar_data.frame, lidar))
   
    # Get the lidar data and convert it to a numpy array.
    p_cloud_size = len(lidar_data)
    print('every frame include '+ str(p_cloud_size)+' points.')
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])
    #print(p_cloud)
    #左手坐标系转换成右手坐标系，这里直接把x变成-x即可
    p_cloud[:,0] = -1 * p_cloud[:,0]
    #print(p_cloud)
    #保存数据
    filename = '_out/lidar/%.6d_point.xyz' % lidar_data.frame
    with open(filename, 'w') as f_obj:
        f_obj.writelines("x\ty\tz\ti\n")
        for i in range(p_cloud_size):
            f_obj.writelines(str(p_cloud[i,0])+" \t "
                            +str(p_cloud[i,1])+" \t "
                            +str(p_cloud[i,2])+" \t "
                            +str(p_cloud[i,3])+"\n")
            
    #lidar_data.save_to_disk('_out/lidar/%.6d.xyz' % lidar_data.frame)  #这样保存下来坐标轴是反着的，和ue4的坐标轴（左手轴）不一样，需要变成右手轴才能看
    lidar_data.save_to_disk('_out/lidar/%.6d.ply' % lidar_data.frame)

    #save vehicle location every frame!
    filename2 = '_out/lidar/vehicle_pos_rot.txt'
    with open(filename2, 'a') as f_obj:
        f_obj.writelines('%.6d'%lidar_data.frame +' '+ str(vehicle.get_transform())+'\n')
    




def main():
    #create client
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

 
    actor_list = []
    sensor_list = []


    try:

        traffic_manager = client.get_trafficmanager()

        # Now let's filter all the blueprints of type 'vehicle' and choose one spawn location
        blueprint_library = world.get_blueprint_library()
        car_bp = blueprint_library.filter('tesla')[1]
        #car_spawnpoint = carla.Transform(carla.Location(x=70.8, y=-7.8, z=0.2), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))#!!!carla coordinate is important for plot!
        car_spawnpoint = carla.Transform(carla.Location(x=27.9, y=-5.6, z=0.2), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
        vehicle = world.spawn_actor(car_bp, car_spawnpoint)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)
        route = [ "Left", "Left", "Left", "Left", "Right"]
        traffic_manager.set_route(vehicle, route)

        

        

        '''
        #spawn a walker and an control walker's movement
        walker_bp = blueprint_library.filter('pedestrian')[1]
        #walker_bp.set_attribute('speed', str(1 + random.random()))
        walker_spawnpoint = carla.Transform(carla.Location(x=63.8, y=-11.5, z=0.2), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000))
        #walker_spawnpoint = carla.Transform(carla.Location(x=50.8, y=-11.5, z=0.2))
        walker = world.spawn_actor(walker_bp, walker_spawnpoint)
        walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=-1, y=0, z=0), speed=1.4, jump=False))
        actor_list.append(walker)
        print('create %s' % walker.type_id)

        '''

        


        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.2
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()


        # Let's add now a "rgb" camera attached to the vehicle. Note that the
        # spawnpoint we give here is now relative to the vehicle.
        # Modify the attributes of the blueprint to set image resolution and field of view.
        # Set the time in seconds between sensor captures,0.0 as fast as possible


        camera_bp = blueprint_library.find('sensor.camera.rgb')  
        camera_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}') #pixels
        camera_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        #camera_bp.set_attribute('sensor_tick', '0.0')
        camera_spawnpoint = carla.Transform(carla.Location(x=1.9, z=1.2))
        camera = world.spawn_actor(camera_bp, camera_spawnpoint, attach_to=vehicle)     
        print('created %s' % camera.type_id)
        camera.listen(lambda image: process_camera(image, sensor_queue, "camera"))
        sensor_list.append(camera)

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # converting the pixels to Raw.No changes applied to the image. Used by the RGB camera.
        
        #camera.listen(lambda image: camera_see(image)) #不能和上面保存图像的命令同时出现，会导致程序错误
       
        #add a lidar attached to the vehicle
        
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(7000000))
        lidar_bp.set_attribute('rotation_frequency',str(10))
        lidar_bp.set_attribute('range',str(30))
        lidar_spawnpoint = carla.Transform(carla.Location(x=1.3,z=1.7)) #lidar has 360° view, so put it on the top of car
        lidar = world.spawn_actor(lidar_bp, lidar_spawnpoint,attach_to=vehicle)
        print('created %s' % lidar.type_id)
        lidar.listen(lambda lidar_data: process_lidar_data(lidar_data, vehicle,sensor_queue, "lidar"))
        sensor_list.append(lidar)


        time.sleep(30)
        
        while True:
            
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            #time.sleep(10)
            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 50.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                
            except Empty:
                print(" Some of the sensor information is missed")
            

        
    finally:
        
        world.apply_settings(original_settings)
        print('destroying actors and sensors!')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()

        print('now all destroyed!')


if __name__ == '__main__':

    main()
