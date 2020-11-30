#Sweeps frame rate: 10Hz

import numpy as np
import cv2 as cv

#NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

class DataLoaderHF:

    def __init__(self, HOME_ROUTE = '/data/sets/nuscenes/', canbus_scenes = 1111, sensor_scenes = 850):
        nusc = NuScenes(version='v1.0-trainval', dataroot=HOME_ROUTE, verbose=True)
        nusc_can = NuScenesCanBus(dataroot=HOME_ROUTE)

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
        #Subtracts the array and the selected timestamp.
        #Returns the index value of the minimun difference.
        def get_closest(list, num):
            list = np.asarray(list)
            id = (np.abs(np.subtract(list,num))).argmin()
            return id

        #Load and save vehicle speed from vehicle_monitor
        scene_name = 'scene-0001'
        message_name = 'vehicle_monitor'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        self.vehicle_speed = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])

        for i in range(2, canbus_scenes):

            scene_name = get_next_scene(i)
            if not scene_name == 'blacklisted':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_speed = np.append(self.vehicle_speed, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)

        #Load and save steering data from vehicle_monitor
        scene_name = 'scene-0001'
        message_name = 'vehicle_monitor'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        self.vehicle_steering = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])

        for i in range(2, canbus_scenes):

            scene_name = get_next_scene(i)
            if not scene_name == 'blacklisted':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_steering = np.append(self.vehicle_steering, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)

        #Load images
        #CAM_FRONT
        sensor = 'CAM_FRONT'

        self.cam_front_array = []
        self.cam_front_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_front_data['next'] == "":
                token = cam_front_data['timestamp']
                route = HOME_ROUTE + str(cam_front_data['filename'])

                self.cam_front_array.append(route)
                self.cam_front_tokens.append(token)

                cam_front_data = nusc.get('sample_data', cam_front_data['next'])

        #Converts the input in an array
        self.cam_front_array = np.asarray(self.cam_front_array)
        self.am_front_tokens = np.asarray(self.cam_front_tokens)

        #CAM_FRONT_LEFT
        sensor = 'CAM_FRONT_LEFT'

        self.cam_front_left_array = []
        self.cam_front_left_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_front_left_data['next'] == "":
                token = cam_front_left_data['timestamp']
                route = HOME_ROUTE + str(cam_front_left_data['filename'])

                self.cam_front_left_array.append(route)
                self.cam_front_left_tokens.append(token)

                cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

        #Converts the input in an array
        self.cam_front_left_array = np.asarray(self.cam_front_left_array)
        self.cam_front_left_tokens = np.asarray(self.cam_front_left_tokens)

        #CAM_FRONT_RIGHT
        sensor = 'CAM_FRONT_RIGHT'

        self.cam_front_right_array = []
        self.cam_front_right_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_front_right_data['next'] == "":
                token = cam_front_right_data['timestamp']
                route = HOME_ROUTE + str(cam_front_right_data['filename'])

                self.cam_front_right_array.append(route)
                self.cam_front_right_tokens.append(token)

                cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])

        #Converts the input in an array
        self.cam_front_right_array = np.asarray(self.cam_front_right_array)
        self.cam_front_right_tokens = np.asarray(self.cam_front_right_tokens)

        #CAM_BACK
        sensor = 'CAM_BACK'

        self.cam_back_array = []
        self.cam_back_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_back_data['next'] == "":
                token = cam_back_data['timestamp']
                route = HOME_ROUTE + str(cam_back_data['filename'])

                self.cam_back_array.append(route)
                self.cam_back_tokens.append(token)

                cam_back_data = nusc.get('sample_data', cam_back_data['next'])

        #Converts the input in an array
        self.cam_back_array = np.asarray(self.cam_back_array)
        self.cam_back_tokens = np.asarray(self.cam_back_tokens)

        #CAM_BACK_LEFT
        sensor = 'CAM_BACK_LEFT'

        self.cam_back_left_array = []
        self.cam_back_left_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_back_left_data['next'] == "":
                token = cam_back_left_data['timestamp']
                route = HOME_ROUTE + str(cam_back_left_data['filename'])

                self.cam_back_left_array.append(route)
                self.cam_back_left_tokens.append(token)

                cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])

        #Converts the input in an array
        self.cam_back_left_array = np.asarray(self.cam_back_left_array)
        self.cam_back_left_tokens = np.asarray(self.cam_back_left_tokens)

        #CAM_BACK_RIGHT
        sensor = 'CAM_BACK_RIGHT'

        self.cam_back_right_array = []
        self.cam_back_right_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            my_sample = nusc.get('sample', my_scene['first_sample_token'])

            cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
            while not cam_back_right_data['next'] == "":
                token = cam_back_right_data['timestamp']
                route = HOME_ROUTE + str(cam_back_right_data['filename'])

                self.cam_back_right_array.append(route)
                self.cam_back_right_tokens.append(token)

                cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])

        #Converts the input in an array
        self.cam_back_right_array = np.asarray(self.cam_back_right_array)
        self.cam_back_right_tokens = np.asarray(self.cam_back_right_tokens)

        #Build the closest arrays
        self.closest_cam_front_array = []
        self.closest_cam_front_tokens = []

        self.closest_cam_front_left_array = []
        self.closest_cam_front_left_tokens = []

        self.closest_cam_front_right_array = []
        self.closest_cam_front_right_tokens = []

        self.closest_cam_back_array = []
        self.closest_cam_back_tokens = []

        self.closest_cam_back_left_array = []
        self.closest_cam_back_left_tokens = []

        self.closest_cam_back_right_array = []
        self.closest_cam_back_right_tokens = []

        for i in range(np.shape(self.vehicle_speed)[0]):

            #CAM FRONT
            id = get_closest(self.cam_front_tokens, self.vehicle_speed[i,0])
            self.closest_cam_front_array.append(self.cam_front_array[id])
            self.closest_cam_front_tokens.append(self.cam_front_tokens[id])

            #CAM FRONT LEFT
            id = get_closest(self.cam_front_left_tokens, self.vehicle_speed[i,0])
            self.closest_cam_front_left_array.append(self.cam_front_left_array[id])
            self.closest_cam_front_left_tokens.append(self.cam_front_left_tokens[id])

            #CAM FRONT RIGHT
            id = get_closest(self.cam_front_right_tokens, self.vehicle_speed[i,0])
            self.closest_cam_front_right_array.append(self.cam_front_right_array[id])
            self.closest_cam_front_right_tokens.append(self.cam_front_right_tokens[id])

            #CAM BACK
            id = get_closest(self.cam_back_tokens, self.vehicle_speed[i,0])
            self.closest_cam_back_array.append(self.cam_back_array[id])
            self.closest_cam_back_tokens.append(self.cam_back_tokens[id])

            #CAM BACK LEFT
            id = get_closest(self.cam_back_left_tokens, self.vehicle_speed[i,0])
            self.closest_cam_back_left_array.append(self.cam_back_left_array[id])
            self.closest_cam_back_left_tokens.append(self.cam_back_left_tokens[id])

            #CAM BACK RIGHT
            id = get_closest(self.cam_back_right_tokens, self.vehicle_speed[i,0])
            self.closest_cam_back_right_array.append(self.cam_back_right_array[id])
            self.closest_cam_back_right_tokens.append(self.cam_back_right_tokens[id])


        self.closest_cam_front_array = np.asarray(self.closest_cam_front_array)
        self.closest_cam_front_tokens = np.asarray(self.closest_cam_front_tokens)

        self.closest_cam_front_left_array = np.asarray(self.closest_cam_front_left_array)
        self.closest_cam_front_left_tokens = np.asarray(self.closest_cam_front_left_tokens)

        self.closest_cam_front_right_array = np.asarray(self.closest_cam_front_right_array)
        self.closest_cam_front_right_tokens = np.asarray(self.closest_cam_front_right_tokens)

        self.closest_cam_back_array = np.asarray(self.closest_cam_back_array)
        self.closest_cam_back_tokens = np.asarray(self.closest_cam_back_tokens)

        self.closest_cam_back_left_array = np.asarray(self.closest_cam_back_left_array)
        self.closest_cam_back_left_tokens = np.asarray(self.closest_cam_back_left_tokens)

        self.closest_cam_back_right_array = np.asarray(self.closest_cam_back_right_array)
        self.closest_cam_back_right_tokens = np.asarray(self.closest_cam_back_right_tokens)


    def show_data(self, sensor = 'CAM_FRONT'):

        if sensor == 'CAM_FRONT':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_front_tokens[i]), cv.imread(self.closest_cam_front_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_tokens[i]))
                    continue
        elif sensor == 'CAM_FRONT_LEFT':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_front_left_tokens[i]), cv.imread(self.closest_cam_front_left_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_left_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_left_tokens[i]))
                    continue
        elif sensor == 'CAM_FRONT_RIGHT':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_front_right_tokens[i]), cv.imread(self.closest_cam_front_right_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_right_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_right_tokens[i]))
                    continue
        elif sensor == 'CAM_BACK':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_back_tokens[i]), cv.imread(self.closest_cam_back_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_back_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_back_tokens[i]))
                    continue
        elif sensor == 'CAM_BACK_LEFT':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_back_left_tokens[i]), cv.imread(self.closest_cam_back_left_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_tokens[i]))
                    continue
        elif sensor == 'CAM_BACK_RIGHT':
            for i in range(np.shape(self.vehicle_speed)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering[i,1]))

                cv.imshow(str(self.closest_cam_back_right_tokens[i]), cv.imread(self.closest_cam_back_right_array[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_back_right_tokens[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_back_right_tokens[i]))
                    continue

    #Getters

    #CAN bus data
    def get_speed_data(self):
        return self.vehicle_speed

    def get_steering_data(self):
        return self.vehicle_steering

    #Image arrays
    def get_cam_front_data(self):
        return self.cam_front_array, self.cam_front_tokens

    def get_cam_front_left_data(self):
        return self.cam_front_left_array, self.cam_front_left_tokens

    def get_cam_front_right_data(self):
        return self.cam_front_right_array, self.cam_front_right_tokens

    def get_cam_back_data(self):
        return self.cam_back_array, self.cam_back_tokens

    def get_cam_back_left_data(self):
        return self.cam_back_left_array, self.cam_back_left_tokens

    def get_cam_back_right_data(self):
        return self.cam_back_right_array, self.cam_back_right_tokens

    #Array containing the closest images to the can bus data
    def get_closest_cam_front_data(self):
        return self.closest_cam_front_array, self.closest_cam_front_tokens

    def get_closest_cam_front_left_data(self):
        return self.closest_cam_front_left_array, self.closest_cam_front_left_tokens

    def get_closest_cam_front_right_data(self):
        return self.closest_cam_front_right_array, self.closest_cam_front_right_tokens

    def get_closest_cam_back_data(self):
        return self.closest_cam_back_array, self.closest_cam_back_tokens

    def get_closest_cam_back_left_data(self):
        return self.closest_cam_back_left_array, self.closest_cam_back_left_tokens

    def get_closest_cam_back_right_data(self):
        return self.closest_cam_back_right_array, self.closest_cam_back_right_tokens
