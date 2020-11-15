#Sweeps frame rate: 10Hz

import numpy as np
import cv2 as cv

HOME_ROUTE = '/home/darjwx/data/sets/nuscenes/'
#nusc_can.can_blacklist has the scenes without can bus data. But there are more scenes that dont exist
#419 does not have vehicle_monitor info
#Missing data for 133 scenes out of 1110 -> aprox 12%
SCENE_BLACKLIST = np.array([ 37, 40, 136, 137, 141, 153, 156, 161, 162, 163,
                            164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                            174, 175, 176, 186, 189, 197, 198, 201, 205, 215,
                            216, 217, 223, 267, 309, 310, 311, 312, 313, 314,
                            319, 320, 322, 325, 326, 327, 387, 404, 409, 419,
                            460, 466, 470, 473, 503, 516, 540, 567, 569, 579,
                            581, 605, 628, 631, 680, 682, 690, 691, 692, 693,
                            694, 699, 702, 720, 721, 722, 723, 724, 725, 729,
                            732, 742, 743, 745, 748, 753, 754, 755, 756, 766,
                            772, 773, 774, 776, 779, 785, 788, 793, 801, 807,
                            814, 818, 823, 824, 825, 826, 832, 843, 857, 859,
                            867, 874, 879, 881, 918, 934, 944, 946, 948, 950,
                            951, 954, 964, 965, 970, 973, 974, 985, 986, 987,
                            993, 1098, 1103])

#NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
nusc = NuScenes(version='v1.0-mini', dataroot=HOME_ROUTE, verbose=True)
nusc_can = NuScenesCanBus(dataroot=HOME_ROUTE)

#Functions

#Receives the number of the scene we want.
#Checks the blacklist to avoid broken scenes
#and returns a formatted string:
#     scene-0001
def get_next_scene(num):
    if num in SCENE_BLACKLIST:
        return 'blacklisted'
    else:
        scene_name = 'scene-' +  str(num).zfill(4)
        return scene_name

#Receives an array with timestamps and the timestamp we want.
#Subtracts the array and the seletcted timestamp.
#Returns the index value of the minimun difference.
def get_closest(list, num):
    list = np.asarray(list)
    id = (np.abs(list - num)).argmin()
    return id


#Load and save vehicle speed from vehicle_monitor
scene_name = 'scene-0001'
#print(scene_name)
message_name = 'vehicle_monitor'
vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
vehicle_speed = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])
#print(vehicle_speed.shape)
#print(vehicle_speed)

scenes = 1111
for i in range(2, scenes):

    scene_name = get_next_scene(i)
    print(scene_name)
    if not scene_name == 'blacklisted':
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        vehicle_speed = np.append(vehicle_speed, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)

    #print(scene_name)
    #print(vehicle_speed.shape)


#np.savetxt('veh_speed.csv', vehicle_speed, delimiter = ',')

#Load and save steering data from vehicle_monitor
scene_name = 'scene-0001'
#print(scene_name)
message_name = 'vehicle_monitor'
vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
vehicle_steering = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])
#print(vehicle_steering)

scenes = 1111
for i in range(2, scenes):

    scene_name = get_next_scene(i)
    if not scene_name == 'blacklisted':
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        vehicle_steering = np.append(vehicle_steering, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)

    #print(scene_name)
    #print(vehicle_speed)


#np.savetxt('veh_steering.csv', vehicle_steering, delimiter = ',')

#Load images
#CAM_FRONT
sensor = 'CAM_FRONT'

scenes = 10
cam_front_array = []
cam_front_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_front_data['next'] == "":
        token = cam_front_data['sample_token']
        route = HOME_ROUTE + str(cam_front_data['filename'])
        #print(route)

        cam_front_array.append(np.array(cv.imread(route)))
        cam_front_tokens.append(token)

        cam_front_data = nusc.get('sample_data', cam_front_data['next'])

        print('sample')


#Converts the input in an array
#print('asarray 1')
cam_front_array = np.asarray(cam_front_array)
#print('asarray 2')

#CAM_FRONT_LEFT
sensor = 'CAM_FRONT_LEFT'

scenes = 10
cam_front_left_array = []
cam_front_left_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_front_left_data['next'] == "":
        token = cam_front_left_data['sample_token']
        route = HOME_ROUTE + str(cam_front_left_data['filename'])

        cam_front_left_array.append(np.array(cv.imread(route)))
        cam_front_left_tokens.append(token)

        cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])
        print('sample')

#Converts the input in an array
cam_front_left_array = np.asarray(cam_front_left_array)

#CAM_FRONT_RIGHT
sensor = 'CAM_FRONT_RIGHT'

scenes = 10
cam_front_right_array = []
cam_front_right_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_front_right_data['next'] == "":
        token = cam_front_right_data['sample_token']
        route = HOME_ROUTE + str(cam_front_right_data['filename'])

        cam_front_right_array.append(np.array(cv.imread(route)))
        cam_front_right_tokens.append(token)

        cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])
        print('sample')

#Converts the input in an array
cam_front_right_array = np.asarray(cam_front_right_array)

#CAM_BACK
sensor = 'CAM_BACK'

scenes = 10
cam_back_array = []
cam_back_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_back_data['next'] == "":
        token = cam_back_data['sample_token']
        route = HOME_ROUTE + str(cam_back_data['filename'])

        cam_back_array.append(np.array(cv.imread(route)))
        cam_back_tokens.append(token)

        cam_back_data = nusc.get('sample_data', cam_back_data['next'])
        print('sample')

#Converts the input in an array
cam_back_array = np.asarray(cam_back_array)

#CAM_BACK_LEFT
sensor = 'CAM_BACK_LEFT'

scenes = 10
cam_back_left_array = []
cam_back_left_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_back_left_data['next'] == "":
        token = cam_back_left_data['sample_token']
        route = HOME_ROUTE + str(cam_back_left_data['filename'])

        cam_back_left_array.append(np.array(cv.imread(route)))
        cam_back_left_tokens.append(token)

        cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])
        print('sample')

#Converts the input in an array
cam_back_left_array = np.asarray(cam_back_left_array)

#CAM_BACK_RIGHT
sensor = 'CAM_BACK_RIGHT'

scenes = 10
cam_back_right_array = []
cam_back_right_tokens = []
for i in range(scenes):
    my_scene = nusc.scene[i]
    my_sample = nusc.get('sample', my_scene['first_sample_token'])

    print('Scene: ' + str(i + 1))
    cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
    while not cam_back_right_data['next'] == "":
        token = cam_back_right_data['sample_token']
        route = HOME_ROUTE + str(cam_back_right_data['filename'])

        cam_back_right_array.append(np.array(cv.imread(route)))
        cam_back_right_tokens.append(token)

        cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])
        print('sample')

#Converts the input in an array
cam_back_right_array = np.asarray(cam_back_right_array)

loop = 1
while loop == 1:
    #Images selection
    im_select = input("Which camara (ex: CAM_FRONT): ")

    if im_select == 'CAM_FRONT':
        for i in range(np.shape(cam_front_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_front_tokens[i], cam_front_array[i])
            my_sample = nusc.get('sample', cam_front_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))


            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue
    elif im_select == 'CAM_FRONT_LEFT':
        for i in range(np.shape(cam_front_left_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_front_left_tokens[i], cam_front_left_array[i])
            my_sample = nusc.get('sample', cam_front_left_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue
    elif im_select == 'CAM_FRONT_RIGHT':
        for i in range(np.shape(cam_front_right_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_front_right_tokens[i], cam_front_right_array[i])
            my_sample = nusc.get('sample', cam_front_right_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue
    elif im_select == 'CAM_BACK':
        for i in range(np.shape(cam_back_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_back_tokens[i], cam_back_array[i])
            my_sample = nusc.get('sample', cam_back_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue
    elif im_select == 'CAM_BACK_LEFT':
        for i in range(np.shape(cam_back_left_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_back_left_tokens[i], cam_back_left_array[i])
            my_sample = nusc.get('sample', cam_back_left_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue
    elif im_select == 'CAM_BACK_RIGHT':
        for i in range(np.shape(cam_back_right_array)[0]):
            #print('Sample: ' + str(i))
            cv.imshow(cam_back_right_tokens[i], cam_back_right_array[i])
            my_sample = nusc.get('sample', cam_back_right_tokens[i])

            #Not sure how efficient this is
            timestamp_sample = my_sample['timestamp']

            id = get_closest(vehicle_speed[:, 0], timestamp_sample)
            print('Vehicle speed ' + str(vehicle_speed[id,1]))

            id = get_closest(vehicle_steering[:, 0], timestamp_sample)
            print('Vehicle steering ' + str(vehicle_steering[id,1]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(cam_front_tokens[i])
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(cam_front_tokens[i])
                continue

    cont = input("Continue (yes/no): ")
    if cont == 'yes':
        continue
    else:
        break

cv.destroyAllWindows()
