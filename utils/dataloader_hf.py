#Sweeps frame rate: 10Hz

import numpy as np
import cv2 as cv

#NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

class DataLoaderHF:

    def __init__(self, HOME_ROUTE = '/data/sets/nuscenes/', canbus_scenes = 1111, sensor_scenes = 850):
        nusc = NuScenes(version='v1.0-trainval', dataroot=HOME_ROUTE, verbose=True)
        nusc_can = NuScenesCanBus(dataroot=HOME_ROUTE)

        #nusc_can.can_blacklist has the scenes without can bus data.
        #Take those scenes and create an array with a compatible format.
        #scene-xxxx
        #419 does not have vehicle_monitor info
        SCENE_BLACKLIST = ['scene-0161', 'scene-0162', 'scene-0163',
                           'scene-0164', 'scene-0165', 'scene-0166',
                           'scene-0167', 'scene-0168', 'scene-0170',
                           'scene-0171', 'scene-0172', 'scene-0173',
                           'scene-0174', 'scene-0175', 'scene-0176',
                           'scene-0309', 'scene-0310', 'scene-0311',
                           'scene-0312', 'scene-0313', 'scene-0314',
                           'scene-0419']

        #Functions

        #Receives the number of the scene we want
        #and the data type. For train, validation or test.
        #Checks the blacklist to avoid broken scenes
        #and returns a formatted string:
        #     scene-0001
        def get_listed_scene(num, scene = None):
            splits = create_splits_scenes()

            if num == -1:
                if scene in splits['train'] and scene not in SCENE_BLACKLIST:
                    return 'train', scene
                elif scene in splits['val'] and scene not in SCENE_BLACKLIST:
                    return 'val', scene
                else:
                    return 'not listed', scene
            else:
                scene_name = 'scene-' +  str(num).zfill(4)

                if scene_name in splits['train'] and scene_name not in SCENE_BLACKLIST:
                    return 'train', scene_name
                elif scene_name in splits['val'] and scene_name not in SCENE_BLACKLIST:
                    return 'val', scene_name
                else:
                    return 'not listed', scene_name

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
        self.vehicle_speed_train = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])
        scene_name = 'scene-0003'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        self.vehicle_speed_val = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])


        for i in range(2, canbus_scenes):
            type, scene_name = get_listed_scene(i)
            if type == 'train':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_speed_train = np.append(self.vehicle_speed_train, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)
            elif type == 'val' and i > 3:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_speed_val = np.append(self.vehicle_speed_val, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)

        #Load and save steering data from vehicle_monitor
        scene_name = 'scene-0001'
        message_name = 'vehicle_monitor'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        self.vehicle_steering_train = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])
        scene_name = 'scene-0003'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        self.vehicle_steering_val = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])

        for i in range(2, canbus_scenes):

            type, scene_name = get_listed_scene(i)
            if type == 'train':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_steering_train = np.append(self.vehicle_steering_train, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)
            elif type == 'val' and i > 3:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                self.vehicle_steering_val = np.append(self.vehicle_steering_val, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)

        #Load images
        #CAM_FRONT
        sensor = 'CAM_FRONT'

        self.cam_front_train = []
        self.cam_front_tokens_train = []
        self.cam_front_val = []
        self.cam_front_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_data['next'] == "":
                    token = cam_front_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_data['filename'])

                    self.cam_front_train.append(route)
                    self.cam_front_tokens_train.append(token)

                    cam_front_data = nusc.get('sample_data', cam_front_data['next'])

            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_data['next'] == "":
                    token = cam_front_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_data['filename'])

                    self.cam_front_val.append(route)
                    self.cam_front_tokens_val.append(token)

                    cam_front_data = nusc.get('sample_data', cam_front_data['next'])

        #Converts the input in an array
        self.cam_front_train = np.asarray(self.cam_front_train)
        self.am_front_tokens = np.asarray(self.cam_front_tokens_train)
        self.cam_front_val = np.asarray(self.cam_front_val)
        self.am_front_tokens_val = np.asarray(self.cam_front_tokens_val)

        #CAM_FRONT_LEFT
        sensor = 'CAM_FRONT_LEFT'

        self.cam_front_left_train = []
        self.cam_front_left_tokens_train = []
        self.cam_front_left_val = []
        self.cam_front_left_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_left_data['next'] == "":
                    token = cam_front_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_left_data['filename'])

                    self.cam_front_left_train.append(route)
                    self.cam_front_left_tokens_train.append(token)

                    cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_left_data['next'] == "":
                    token = cam_front_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_left_data['filename'])

                    self.cam_front_left_val.append(route)
                    self.cam_front_left_tokens_val.append(token)

                    cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

        #Converts the input in an array
        self.cam_front_left_train = np.asarray(self.cam_front_left_train)
        self.cam_front_left_tokens_train = np.asarray(self.cam_front_left_tokens_train)
        self.cam_front_left_val = np.asarray(self.cam_front_left_val)
        self.cam_front_left_tokens_val = np.asarray(self.cam_front_left_tokens_val)

        #CAM_FRONT_RIGHT
        sensor = 'CAM_FRONT_RIGHT'

        self.cam_front_right_train = []
        self.cam_front_right_tokens_train = []
        self.cam_front_right_val = []
        self.cam_front_right_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_right_data['next'] == "":
                    token = cam_front_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_right_data['filename'])

                    self.cam_front_right_train.append(route)
                    self.cam_front_right_tokens_train.append(token)

                    cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_right_data['next'] == "":
                    token = cam_front_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_right_data['filename'])

                    self.cam_front_right_val.append(route)
                    self.cam_front_right_tokens_val.append(token)

                    cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])


        #Converts the input in an array
        self.cam_front_right_train = np.asarray(self.cam_front_right_train)
        self.cam_front_right_tokens_train = np.asarray(self.cam_front_right_tokens_train)
        self.cam_front_right_val = np.asarray(self.cam_front_right_val)
        self.cam_front_right_tokens_val = np.asarray(self.cam_front_right_tokens_val)

        #CAM_BACK
        sensor = 'CAM_BACK'

        self.cam_back_train = []
        self.cam_back_tokens_train = []
        self.cam_back_val = []
        self.cam_back_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_data['next'] == "":
                    token = cam_back_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_data['filename'])

                    self.cam_back_train.append(route)
                    self.cam_back_tokens_train.append(token)

                    cam_back_data = nusc.get('sample_data', cam_back_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_data['next'] == "":
                    token = cam_back_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_data['filename'])

                    self.cam_back_val.append(route)
                    self.cam_back_tokens_val.append(token)

                    cam_back_data = nusc.get('sample_data', cam_back_data['next'])


        #Converts the input in an array
        self.cam_back_train = np.asarray(self.cam_back_train)
        self.cam_back_tokens_train = np.asarray(self.cam_back_tokens_train)
        self.cam_back_val = np.asarray(self.cam_back_val)
        self.cam_back_tokens_val = np.asarray(self.cam_back_tokens_val)


        #CAM_BACK_LEFT
        sensor = 'CAM_BACK_LEFT'

        self.cam_back_left_train = []
        self.cam_back_left_tokens_train = []
        self.cam_back_left_val = []
        self.cam_back_left_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_left_data['next'] == "":
                    token = cam_back_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_left_data['filename'])

                    self.cam_back_left_train.append(route)
                    self.cam_back_left_tokens_train.append(token)

                    cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_left_data['next'] == "":
                    token = cam_back_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_left_data['filename'])

                    self.cam_back_left_val.append(route)
                    self.cam_back_left_tokens_val.append(token)

                    cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])


        #Converts the input in an array
        self.cam_back_left_train = np.asarray(self.cam_back_left_train)
        self.cam_back_left_tokens_train = np.asarray(self.cam_back_left_tokens_train)
        self.cam_back_left_val = np.asarray(self.cam_back_left_val)
        self.cam_back_left_tokens_val = np.asarray(self.cam_back_left_tokens_val)

        #CAM_BACK_RIGHT
        sensor = 'CAM_BACK_RIGHT'

        self.cam_back_right_train = []
        self.cam_back_right_tokens_train = []
        self.cam_back_right_val = []
        self.cam_back_right_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_right_data['next'] == "":
                    token = cam_back_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_right_data['filename'])

                    self.cam_back_right_train.append(route)
                    self.cam_back_right_tokens_train.append(token)

                    cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_right_data['next'] == "":
                    token = cam_back_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_right_data['filename'])

                    self.cam_back_right_val.append(route)
                    self.cam_back_right_tokens_val.append(token)

                    cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])


        #Converts the input in an array
        self.cam_back_right_train = np.asarray(self.cam_back_right_train)
        self.cam_back_right_tokens_train = np.asarray(self.cam_back_right_tokens_train)
        self.cam_back_right_val = np.asarray(self.cam_back_right_val)
        self.cam_back_right_tokens_val = np.asarray(self.cam_back_right_tokens_val)

        #Build the closest arrays
        self.closest_cam_front_train = []
        self.closest_cam_front_tokens_train = []

        self.closest_cam_front_val = []
        self.closest_cam_front_tokens_val = []

        self.closest_cam_front_left_train = []
        self.closest_cam_front_left_tokens_train = []

        self.closest_cam_front_left_val = []
        self.closest_cam_front_left_tokens_val = []

        self.closest_cam_front_right_train = []
        self.closest_cam_front_right_tokens_train = []

        self.closest_cam_front_right_val = []
        self.closest_cam_front_right_tokens_val = []

        self.closest_cam_back_train = []
        self.closest_cam_back_tokens_train = []

        self.closest_cam_back_val = []
        self.closest_cam_back_tokens_val = []

        self.closest_cam_back_left_train = []
        self.closest_cam_back_left_tokens_train = []

        self.closest_cam_back_left_val = []
        self.closest_cam_back_left_tokens_val = []

        self.closest_cam_back_right_train = []
        self.closest_cam_back_right_tokens_train = []

        self.closest_cam_back_right_val = []
        self.closest_cam_back_right_tokens_val = []


        for i in range(np.shape(self.vehicle_speed_train)[0]):

            print(i)
            #CAM FRONT
            id = get_closest(self.cam_front_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_front_train.append(self.cam_front_train[id])
            self.closest_cam_front_tokens_train.append(self.cam_front_tokens_train[id])

            #CAM FRONT LEFT
            id = get_closest(self.cam_front_left_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_front_left_train.append(self.cam_front_left_train[id])
            self.closest_cam_front_left_tokens_train.append(self.cam_front_left_tokens_train[id])

            #CAM FRONT RIGHT
            id = get_closest(self.cam_front_right_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_front_right_train.append(self.cam_front_right_train[id])
            self.closest_cam_front_right_tokens_train.append(self.cam_front_right_tokens_train[id])

            #CAM BACK
            id = get_closest(self.cam_back_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_back_train.append(self.cam_back_train[id])
            self.closest_cam_back_tokens_train.append(self.cam_back_tokens_train[id])

            #CAM BACK LEFT
            id = get_closest(self.cam_back_left_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_back_left_train.append(self.cam_back_left_train[id])
            self.closest_cam_back_left_tokens_train.append(self.cam_back_left_tokens_train[id])

            #CAM BACK RIGHT
            id = get_closest(self.cam_back_right_tokens_train, self.vehicle_speed_train[i,0])
            self.closest_cam_back_right_train.append(self.cam_back_right_train[id])
            self.closest_cam_back_right_tokens_train.append(self.cam_back_right_tokens_train[id])

        for i in range(np.shape(self.vehicle_speed_val)[0]):


            id = get_closest(self.cam_front_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_front_val.append(self.cam_front_val[id])
            self.closest_cam_front_tokens_val.append(self.cam_front_tokens_val[id])

            id = get_closest(self.cam_front_left_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_front_left_val.append(self.cam_front_left_val[id])
            self.closest_cam_front_left_tokens_val.append(self.cam_front_left_tokens_val[id])

            id = get_closest(self.cam_front_right_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_front_right_val.append(self.cam_front_right_val[id])
            self.closest_cam_front_right_tokens_val.append(self.cam_front_right_tokens_val[id])

            id = get_closest(self.cam_back_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_back_val.append(self.cam_back_val[id])
            self.closest_cam_back_tokens_val.append(self.cam_back_tokens_val[id])

            id = get_closest(self.cam_back_left_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_back_left_val.append(self.cam_back_left_val[id])
            self.closest_cam_back_left_tokens_val.append(self.cam_back_left_tokens_val[id])

            id = get_closest(self.cam_back_right_tokens_val, self.vehicle_speed_val[i,0])
            self.closest_cam_back_right_val.append(self.cam_back_right_val[id])
            self.closest_cam_back_right_tokens_val.append(self.cam_back_right_tokens_val[id])

        self.closest_cam_front_train = np.asarray(self.closest_cam_front_train)
        self.closest_cam_front_tokens_train = np.asarray(self.closest_cam_front_tokens_train)

        self.closest_cam_front_val = np.asarray(self.closest_cam_front_val)
        self.closest_cam_front_tokens_val = np.asarray(self.closest_cam_front_tokens_val)

        self.closest_cam_front_left_train = np.asarray(self.closest_cam_front_left_train)
        self.closest_cam_front_left_tokens_train = np.asarray(self.closest_cam_front_left_tokens_train)

        self.closest_cam_front_left_val = np.asarray(self.closest_cam_front_left_val)
        self.closest_cam_front_left_tokens_val = np.asarray(self.closest_cam_front_left_tokens_val)

        self.closest_cam_front_right_train = np.asarray(self.closest_cam_front_right_train)
        self.closest_cam_front_right_tokens_train = np.asarray(self.closest_cam_front_right_tokens_train)

        self.closest_cam_front_right_val = np.asarray(self.closest_cam_front_right_val)
        self.closest_cam_front_right_tokens_val = np.asarray(self.closest_cam_front_right_tokens_val)

        self.closest_cam_back_train = np.asarray(self.closest_cam_back_train)
        self.closest_cam_back_tokens_train = np.asarray(self.closest_cam_back_tokens_train)

        self.closest_cam_back_val = np.asarray(self.closest_cam_back_val)
        self.closest_cam_back_tokens_val = np.asarray(self.closest_cam_back_tokens_val)

        self.closest_cam_back_left_train = np.asarray(self.closest_cam_back_left_train)
        self.closest_cam_back_left_tokens_train = np.asarray(self.closest_cam_back_left_tokens_train)

        self.closest_cam_back_left_val = np.asarray(self.closest_cam_back_left_val)
        self.closest_cam_back_left_tokens_val = np.asarray(self.closest_cam_back_left_tokens_val)

        self.closest_cam_back_right_train = np.asarray(self.closest_cam_back_right_train)
        self.closest_cam_back_right_tokens_train = np.asarray(self.closest_cam_back_right_tokens_train)

        self.closest_cam_back_right_val = np.asarray(self.closest_cam_back_right_val)
        self.closest_cam_back_right_tokens_val = np.asarray(self.closest_cam_back_right_tokens_val)


    def show_data(self, sensor = 'CAM_FRONT', labels = False, labels_array = None):

        if sensor == 'CAM_FRONT':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_front_tokens_train[i]), cv.imread(self.closest_cam_front_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_tokens_train[i]))
                    continue
        elif sensor == 'CAM_FRONT_LEFT':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_front_left_tokens_train[i]), cv.imread(self.closest_cam_front_left_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_left_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_left_tokens_train[i]))
                    continue
        elif sensor == 'CAM_FRONT_RIGHT':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_front_right_tokens_train[i]), cv.imread(self.closest_cam_front_right_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_right_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_right_tokens_train[i]))
                    continue
        elif sensor == 'CAM_BACK':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_back_tokens_train[i]), cv.imread(self.closest_cam_back_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_back_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_back_tokens_train[i]))
                    continue
        elif sensor == 'CAM_BACK_LEFT':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_back_left_tokens_train[i]), cv.imread(self.closest_cam_back_left_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_front_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_front_tokens_train[i]))
                    continue
        elif sensor == 'CAM_BACK_RIGHT':
            for i in range(np.shape(self.vehicle_speed_train)[0]):

                print('Vehicle speed ' + str(self.vehicle_speed_train[i,1]))
                print('Vehicle steering ' + str(self.vehicle_steering_train[i,1]))

                if labels == True:
                    print('Direccion: ' + labels_array[i,0] + ' Acelerador: ' + labels_array[i,1])

                cv.imshow(str(self.closest_cam_back_right_tokens_train[i]), cv.imread(self.closest_cam_back_right_train[i]))

                key = cv.waitKey(0)

                #ESCAPE key
                if key == 27:
                    cv.destroyWindow(str(self.closest_cam_back_right_tokens_train[i]))
                    break
                #ENTER key
                elif key == 13:
                    cv.destroyWindow(str(self.closest_cam_back_right_tokens_train[i]))
                    continue

    #Build labels
    #[i,0] straight, left, right.
    #[i,1] stop, accel, stoping.
    #We consider actual and previous data.
    def get_labels(self):

        labels_array = np.empty(np.shape(self.vehicle_speed_train), dtype = object)
        for i in range(np.shape(self.vehicle_speed_train)[0]):
            #First position does not have previous data
            if i == 0:
                diff_speed = self.vehicle_speed_train[i,1]
                diff_steering = self.vehicle_steering_train[i,1]

                if self.vehicle_steering_train[i,1] > 0:
                    #Left
                    steering = 1
                else:
                    #Right
                    steering = 0
            else:
                diff_speed = self.vehicle_speed_train[i,1] - self.vehicle_speed_train[i-1,1]
                diff_steering = self.vehicle_steering_train[i,1] - self.vehicle_steering_train[i-1,1]

                if self.vehicle_steering_train[i,1] > self.vehicle_steering_train[i-1,1]:
                    #Left
                    steering = 1
                else:
                    #Right
                    steering = 0

            if self.vehicle_speed_train[i,1] == 0:
                labels_array[i,1] = 'stop'
            elif diff_speed > 0:
                labels_array[i,1] = 'accel'
            else:
                labels_array[i,1] = 'stoping'

            if abs(diff_steering) > 0.6 and steering == 1:
                labels_array[i,0] = 'left'
            elif abs(diff_steering) > 0.6 and steering == 0:
                labels_array[i,0] = 'right'
            else:
                labels_array[i,0] = 'straight'

        return labels_array

    #Getters

    #CAN bus data
    def get_canbus_data(self, data = 'speed', type = 'train'):
        if type == 'train':
            if data == 'speed':
                return self.vehicle_speed_train
            else:
                return self.vehicle_steering_train

        elif type == 'val':
            if data == 'speed':
                return self.vehicle_speed_val
            else:
                return self.vehicle_steering_val

    #Image arrays
    def get_sensors_data(self, sensor = 'CAM_FRONT', type = 'train'):
        if type == 'train':
            if sensor == 'CAM_FRONT':
                return self.cam_front_train, self.cam_front_tokens_train

            elif sensor == 'CAM_FRONT_LEFT':
                return self.cam_front_left_train, self.cam_front_left_tokens_train

            elif sensor == 'CAM_FRONT_RIGHT':
                return self.cam_front_right_train, self.cam_front_right_tokens_train

            elif sensor == 'CAM_BACK':
                return self.cam_back_train, self.cam_back_tokens_train

            elif sensor == 'CAM_BACK_LEFT':
                return self.cam_back_left_train, self.cam_back_left_tokens_train

            elif sensor == 'CAM_BACK_RIGHT':
                return self.cam_back_right_train, self.cam_back_right_tokens_train

        elif type == 'val':
            if sensor == 'CAM_FRONT':
                return self.cam_front_val, self.cam_front_tokens_val

            elif sensor == 'CAM_FRONT_LEFT':
                return self.cam_front_left_val, self.cam_front_left_tokens_val

            elif sensor == 'CAM_FRONT_RIGHT':
                return self.cam_front_right_val, self.cam_front_right_tokens_val

            elif sensor == 'CAM_BACK':
                return self.cam_back_val, self.cam_back_tokens_val

            elif sensor == 'CAM_BACK_LEFT':
                return self.cam_back_left_val, self.cam_back_left_tokens_val

            elif sensor == 'CAM_BACK_RIGHT':
                return self.cam_back_right_val, self.cam_back_right_tokens_val


    #Array containing the closest images to the can bus data
    def get_closest_sensors_data(self, sensor = 'CAM_FRONT', type = 'train'):
        if type == 'train':
            if sensor == 'CAM_FRONT':
                return self.closest_cam_front_train, self.closest_cam_front_tokens

            elif sensor == 'CAM_FRONT_LEFT':
                return self.closest_cam_front_left_train, self.closest_cam_front_left_tokens_train

            elif sensor == 'CAM_FRONT_RIGHT':
                return self.closest_cam_front_right_train, self.closest_cam_front_right_tokens_train

            elif sensor == 'CAM_BACK':
                return self.closest_cam_back_train, self.closest_cam_back_tokens_train

            elif sensor == 'CAM_BACK_LEFT':
                return self.closest_cam_back_left_train, self.closest_cam_back_left_tokens_train

            elif sensor == 'CAM_BACK_RIGHT':
                return self.closest_cam_back_right_train, self.closest_cam_back_right_tokens_train

        elif type == 'val':
            if sensor == 'CAM_FRONT':
                return self.closest_cam_front_val, self.closest_cam_front_tokens_val

            elif sensor == 'CAM_FRONT_LEFT':
                return self.closest_cam_front_left_val, self.closest_cam_front_left_tokens_val

            elif sensor == 'CAM_FRONT_RIGHT':
                return self.closest_cam_front_right_val, self.closest_cam_front_right_tokens_val

            elif sensor == 'CAM_BACK':
                return self.closest_cam_back_val, self.closest_cam_back_tokens_val

            elif sensor == 'CAM_BACK_LEFT':
                return self.closest_cam_back_left_val, self.closest_cam_back_left_tokens_val

            elif sensor == 'CAM_BACK_RIGHT':
                return self.closest_cam_back_right_val, self.closest_cam_back_right_tokens_val
