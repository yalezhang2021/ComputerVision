# lesen vehicle position and rotaiton from txt file
# and save them as a better view, frame, xyz, rotxyz
import numpy as np
import re
input_path = ""
output_path = ""

x = []
y = []
z = []
rot_x = []
rot_y = []
rot_z = []
frame = []
filename = "vehicle_pos_rot.txt"
#pos_rot = np.loadtxt(filename,dtype=str)
with open(filename) as f:
    for line in f:
        find_numbers = re.findall(r"\d+\.?\d*|-\d+\.?\d*", line) #list
        frame.append(find_numbers[0])
        x.append(find_numbers[1])
        y.append(find_numbers[2])
        z.append(find_numbers[3])
        rot_y.append(find_numbers[4])
        rot_z.append(find_numbers[5])
        rot_x.append(find_numbers[6])

filename2 = "frame_pos_rot.txt"
with open(filename2,'a') as f_obj:
    f_obj.writelines("frame x y z pitch-Y yaw-Z roll-X\n")
    for i in range(len(frame)):
        f_obj.writelines(frame[i]+' '+x[i]+' '+ y[i]+' '+ z[i]+' '+rot_x[i]+' '+rot_y[i]+' '+rot_z[i]+'\n')
    

#x['frame','x', 'y','z', 'pitch-Y','yaw-Z', 'roll-X']
# rotation in degrees
# position in meters
#     


