#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
world = None

def process_image(image):
    """
    Callback method for camera sensor
    saves camera data as png files
    """
    file_directory = 'data/camera'
    if not os.path.exists(file_directory):
        os.mkdir(file_directory)

    filename = os.path.join(file_directory, 'camera_image_{}.png'.format(image.frame))
    image.save_to_disk(filename)

def render_radar(radar_data):
    """
    Alternativ call back for radar data. Draws radar data directly
    into the UE4 View
    """ 
    velocity_range = 7.5 # m/s
    current_rot = radar_data.transform.rotation
    for detect in radar_data:
        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)
        # The 0.25 adjusts a bit the distance so the dots can
        # be properly seen
        fw_vec = carla.Vector3D(x=detect.depth - 0.25)
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=current_rot.yaw + azi,
                roll=current_rot.roll)).transform(fw_vec)

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))

        norm_velocity = detect.velocity / velocity_range # range [-1, 1]
        r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
        g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
        b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
        world.debug.draw_point(
            radar_data.transform.location + fw_vec,
            size=0.075,
            life_time=0.06,
            persistent_lines=False,
            color=carla.Color(r, g, b))

def rad_callback(radar_data):
    """
    This methods reads out the radar detection and saves them
    as scatter plots
    """

    x_positions = np.empty(len(radar_data))
    y_positions = np.empty(len(radar_data))
    for i, detection in enumerate(radar_data):
        # convert to x and y
        x_pos = detection.depth*np.cos((-detection.azimuth + np.pi/2.0))
        y_pos = detection.depth*np.sin((-detection.azimuth + np.pi/2.0))

        x_positions[i] = x_pos
        y_positions[i] = y_pos

    plt.figure()
    plt.xlim(-60, 60)
    plt.ylim(0, 40)
    plt.scatter(x_positions, y_positions)
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')

    file_directory = 'data/radar'
    if not os.path.exists(file_directory):
        os.mkdir(file_directory)

    filename = os.path.join(file_directory, 'radar_detection_{}.png'.format(radar_data.frame))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    actor_list = []

    try:
        global world
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # create car start position        
        car_spawn_point = world.get_map().get_spawn_points()[0]
        car_offset_point = carla.Location(x=0.0, y=0.0, z=0.0)
        car_spawn_location = car_spawn_point.location
        car_final_location = car_offset_point + car_spawn_location
        car_total_transform = carla.Transform(car_final_location)

        # create car with acceleration of 1
        car_bp = blueprint_library.filter("bmw")[0]
        vehicle = world.spawn_actor(car_bp, car_total_transform)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        actor_list.append(vehicle)

        # create the camera sensor
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        cam_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        cam_bp.set_attribute('fov', "110")
        
        # attach camera to car
        camera_spawn_point = carla.Transform(carla.Location(x=2.0, z=1.0))
        camera_sensor = world.spawn_actor(cam_bp, camera_spawn_point, attach_to=vehicle)
        camera_sensor.listen(lambda data: process_image(data))
        actor_list.append(camera_sensor)

        # create radar sensor
        radar_bp = blueprint_library.find("sensor.other.radar")
        radar_bp.set_attribute('horizontal_fov', str(110))
        radar_bp.set_attribute('vertical_fov', str(30))
        radar_bp.set_attribute('range', str(40))

        # attach radar to car
        rad_location = carla.Location(x=2.0, z=1.0)
        rad_rotation = carla.Rotation(pitch=5.0)
        radar_spawn_point = carla.Transform(rad_location, rad_rotation)
        radar_sensor = world.spawn_actor(radar_bp,radar_spawn_point,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        radar_sensor.listen(lambda radar_data: rad_callback(radar_data))
        actor_list.append(radar_sensor)

        # run simulation for 30 seconds
        time.sleep(30)

    finally:
        print('destroying actors')

        for actor in actor_list:
            actor.destroy()
        print('done.')

if __name__ == '__main__':

    main()
