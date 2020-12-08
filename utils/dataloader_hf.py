#Sweeps frame rate: 10Hz

import numpy as np
import cv2 as cv

#NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

#Progress bar
from tqdm import tqdm

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
        vehicle_speed_train = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])
        scene_name = 'scene-0003'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        vehicle_speed_val = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])


        for i in range(2, canbus_scenes):
            type, scene_name = get_listed_scene(i)
            if type == 'train':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_speed_train = np.append(vehicle_speed_train, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)
            elif type == 'val' and i > 3:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_speed_val = np.append(vehicle_speed_val, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)

        #Load and save steering data from vehicle_monitor
        scene_name = 'scene-0001'
        message_name = 'vehicle_monitor'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        vehicle_steering_train = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])
        scene_name = 'scene-0003'
        vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
        vehicle_steering_val = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])

        for i in range(2, canbus_scenes):

            type, scene_name = get_listed_scene(i)
            if type == 'train':
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_steering_train = np.append(vehicle_steering_train, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)
            elif type == 'val' and i > 3:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_steering_val = np.append(vehicle_steering_val, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)

        #Dictionaries
        self.can_bus = {
            'train': {
                'speed': vehicle_speed_train,
                'steering': vehicle_steering_train
            },
            'val': {
                'speed': vehicle_speed_val,
                'steering': vehicle_steering_val
            }
        }

        #Load images
        #CAM_FRONT
        sensor = 'CAM_FRONT'

        cam_front_train = []
        cam_front_tokens_train = []
        cam_front_val = []
        cam_front_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_data['next'] == "":
                    token = cam_front_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_data['filename'])

                    cam_front_train.append(route)
                    cam_front_tokens_train.append(token)

                    cam_front_data = nusc.get('sample_data', cam_front_data['next'])

            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_data['next'] == "":
                    token = cam_front_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_data['filename'])

                    cam_front_val.append(route)
                    cam_front_tokens_val.append(token)

                    cam_front_data = nusc.get('sample_data', cam_front_data['next'])

        #Converts the input in an array
        cam_front_train = np.asarray(cam_front_train)
        cam_front_tokens_train = np.asarray(cam_front_tokens_train)
        cam_front_val = np.asarray(cam_front_val)
        cam_front_tokens_val = np.asarray(cam_front_tokens_val)

        #CAM_FRONT_LEFT
        sensor = 'CAM_FRONT_LEFT'

        cam_front_left_train = []
        cam_front_left_tokens_train = []
        cam_front_left_val = []
        cam_front_left_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_left_data['next'] == "":
                    token = cam_front_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_left_data['filename'])

                    cam_front_left_train.append(route)
                    cam_front_left_tokens_train.append(token)

                    cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_left_data['next'] == "":
                    token = cam_front_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_left_data['filename'])

                    cam_front_left_val.append(route)
                    cam_front_left_tokens_val.append(token)

                    cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

        #Converts the input in an array
        cam_front_left_train = np.asarray(cam_front_left_train)
        cam_front_left_tokens_train = np.asarray(cam_front_left_tokens_train)
        cam_front_left_val = np.asarray(cam_front_left_val)
        cam_front_left_tokens_val = np.asarray(cam_front_left_tokens_val)


        #CAM_FRONT_RIGHT
        sensor = 'CAM_FRONT_RIGHT'

        cam_front_right_train = []
        cam_front_right_tokens_train = []
        cam_front_right_val = []
        cam_front_right_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_right_data['next'] == "":
                    token = cam_front_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_right_data['filename'])

                    cam_front_right_train.append(route)
                    cam_front_right_tokens_train.append(token)

                    cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_right_data['next'] == "":
                    token = cam_front_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_right_data['filename'])

                    cam_front_right_val.append(route)
                    cam_front_right_tokens_val.append(token)

                    cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])


        #Converts the input in an array
        cam_front_right_train = np.asarray(cam_front_right_train)
        cam_front_right_tokens_train = np.asarray(cam_front_right_tokens_train)
        cam_front_right_val = np.asarray(cam_front_right_val)
        cam_front_right_tokens_val = np.asarray(cam_front_right_tokens_val)

        #CAM_BACK
        sensor = 'CAM_BACK'

        cam_back_train = []
        cam_back_tokens_train = []
        cam_back_val = []
        cam_back_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_data['next'] == "":
                    token = cam_back_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_data['filename'])

                    cam_back_train.append(route)
                    cam_back_tokens_train.append(token)

                    cam_back_data = nusc.get('sample_data', cam_back_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_data['next'] == "":
                    token = cam_back_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_data['filename'])

                    cam_back_val.append(route)
                    cam_back_tokens_val.append(token)

                    cam_back_data = nusc.get('sample_data', cam_back_data['next'])


        #Converts the input in an array
        cam_back_train = np.asarray(cam_back_train)
        cam_back_tokens_train = np.asarray(cam_back_tokens_train)
        cam_back_val = np.asarray(cam_back_val)
        cam_back_tokens_val = np.asarray(cam_back_tokens_val)


        #CAM_BACK_LEFT
        sensor = 'CAM_BACK_LEFT'

        cam_back_left_train = []
        cam_back_left_tokens_train = []
        cam_back_left_val = []
        cam_back_left_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_left_data['next'] == "":
                    token = cam_back_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_left_data['filename'])

                    cam_back_left_train.append(route)
                    cam_back_left_tokens_train.append(token)

                    cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_left_data['next'] == "":
                    token = cam_back_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_left_data['filename'])

                    cam_back_left_val.append(route)
                    cam_back_left_tokens_val.append(token)

                    cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])


        #Converts the input in an array
        cam_back_left_train = np.asarray(cam_back_left_train)
        cam_back_left_tokens_train = np.asarray(cam_back_left_tokens_train)
        cam_back_left_val = np.asarray(cam_back_left_val)
        cam_back_left_tokens_val = np.asarray(cam_back_left_tokens_val)

        #CAM_BACK_RIGHT
        sensor = 'CAM_BACK_RIGHT'

        cam_back_right_train = []
        cam_back_right_tokens_train = []
        cam_back_right_val = []
        cam_back_right_tokens_val = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == 'train':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_right_data['next'] == "":
                    token = cam_back_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_right_data['filename'])

                    cam_back_right_train.append(route)
                    cam_back_right_tokens_train.append(token)

                    cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])
            elif type == 'val':
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_right_data['next'] == "":
                    token = cam_back_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_right_data['filename'])

                    cam_back_right_val.append(route)
                    cam_back_right_tokens_val.append(token)

                    cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])


        #Converts the input in an array
        cam_back_right_train = np.asarray(cam_back_right_train)
        cam_back_right_tokens_train = np.asarray(cam_back_right_tokens_train)
        cam_back_right_val = np.asarray(cam_back_right_val)
        cam_back_right_tokens_val = np.asarray(cam_back_right_tokens_val)

        #Dictionaries
        self.sensors_train = {
            'images': {
                'CAM_FRONT': cam_front_train,
                'CAM_FRONT_LEFT': cam_front_left_train,
                'CAM_FRONT_RIGHT': cam_front_right_train,
                'CAM_BACK': cam_back_train,
                'CAM_BACK_LEFT': cam_back_left_train,
                'CAM_BACK_RIGHT': cam_back_right_train
            },
            'tokens':{
                'CAM_FRONT': cam_front_tokens_train,
                'CAM_FRONT_LEFT': cam_front_left_tokens_train,
                'CAM_FRONT_RIGHT': cam_front_right_tokens_train,
                'CAM_BACK': cam_back_tokens_train,
                'CAM_BACK_LEFT': cam_back_left_tokens_train,
                'CAM_BACK_RIGHT': cam_back_right_tokens_train
            }
        }

        self.sensors_val = {
            'images': {
                'CAM_FRONT': cam_front_val,
                'CAM_FRONT_LEFT': cam_front_left_val,
                'CAM_FRONT_RIGHT': cam_front_right_val,
                'CAM_BACK': cam_back_val,
                'CAM_BACK_LEFT': cam_back_left_val,
                'CAM_BACK_RIGHT': cam_back_right_val
            },
            'tokens': {
                'CAM_FRONT': cam_front_tokens_val,
                'CAM_FRONT_LEFT': cam_front_left_tokens_val,
                'CAM_FRONT_RIGHT': cam_front_right_tokens_val,
                'CAM_BACK': cam_back_tokens_val,
                'CAM_BACK_LEFT': cam_back_left_tokens_val,
                'CAM_BACK_RIGHT': cam_back_right_tokens_val
            }
        }

        #Build the closest arrays
        aux_canbus_train = self.can_bus['train']['speed']
        aux_canbus_val = self.can_bus['val']['speed']

        closest_cam_front_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.dtype('<U122'))
        closest_cam_front_tokens_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.int64)

        closest_cam_front_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_front_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)

        closest_cam_front_left_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.dtype('<U122'))
        closest_cam_front_left_tokens_train = np.empty(np.shape(aux_canbus_train), dtype=np.int64)

        closest_cam_front_left_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_front_left_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)

        closest_cam_front_right_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.dtype('<U122'))
        closest_cam_front_right_tokens_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.int64)

        closest_cam_front_right_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_front_right_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)

        closest_cam_back_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.dtype('<U122'))
        closest_cam_back_tokens_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.int64)

        closest_cam_back_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_back_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)

        closest_cam_back_left_train = np.empty(np.shape(aux_canbus_train), dtype=np.dtype('<U122'))
        closest_cam_back_left_tokens_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.int64)

        closest_cam_back_left_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_back_left_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)

        closest_cam_back_right_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.dtype('<U122'))
        closest_cam_back_right_tokens_train = np.empty(np.shape(aux_canbus_train)[0], dtype=np.int64)

        closest_cam_back_right_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.dtype('<U122'))
        closest_cam_back_right_tokens_val = np.empty(np.shape(aux_canbus_val)[0], dtype=np.int64)


        print('Building train closest arrays')
        for i in tqdm(range(np.shape(aux_canbus_train)[0])):

            #CAM FRONT
            aux_list = self.sensors_train['images']['CAM_FRONT']
            aux_list_tokens = self.sensors_train['tokens']['CAM_FRONT']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_front_train[i] = aux_list[id]
            closest_cam_front_tokens_train[i] = aux_list_tokens[id]

            #CAM FRONT LEFT
            aux_list = self.sensors_train['images']['CAM_FRONT_LEFT']
            aux_list_tokens = self.sensors_train['tokens']['CAM_FRONT_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_front_left_train[i] = aux_list[id]
            closest_cam_front_left_tokens_train[i] = aux_list_tokens[id]

            #CAM FRONT RIGHT
            aux_list = self.sensors_train['images']['CAM_FRONT_RIGHT']
            aux_list_tokens = self.sensors_train['tokens']['CAM_FRONT_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_front_right_train[i] = aux_list[id]
            closest_cam_front_right_tokens_train[i] = aux_list_tokens[id]

            #CAM BACK
            aux_list = self.sensors_train['images']['CAM_BACK']
            aux_list_tokens = self.sensors_train['tokens']['CAM_BACK']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_back_train[i] = aux_list[id]
            closest_cam_back_tokens_train[i] = aux_list_tokens[id]

            #CAM BACK LEFT
            aux_list = self.sensors_train['images']['CAM_BACK_LEFT']
            aux_list_tokens = self.sensors_train['tokens']['CAM_BACK_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_back_left_train[i] = aux_list[id]
            closest_cam_back_left_tokens_train[i] = aux_list_tokens[id]

            #CAM BACK RIGHT
            aux_list = self.sensors_train['images']['CAM_BACK_RIGHT']
            aux_list_tokens = self.sensors_train['tokens']['CAM_BACK_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus_train[i,0])
            closest_cam_back_right_train[i] = aux_list[id]
            closest_cam_back_right_tokens_train[i] = aux_list_tokens[id]

        print('Building validation closest arrays')
        for i in tqdm(range(np.shape(aux_canbus_val)[0])):

            aux_list = self.sensors_val['images']['CAM_FRONT']
            aux_list_tokens = self.sensors_val['tokens']['CAM_FRONT']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_front_val[i] = aux_list[id]
            closest_cam_front_tokens_val[i] = aux_list_tokens[id]

            aux_list = self.sensors_val['images']['CAM_FRONT_LEFT']
            aux_list_tokens = self.sensors_val['tokens']['CAM_FRONT_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_front_left_val[i] = aux_list[id]
            closest_cam_front_left_tokens_val[i] = aux_list_tokens[id]

            aux_list = self.sensors_val['images']['CAM_FRONT_RIGHT']
            aux_list_tokens = self.sensors_val['tokens']['CAM_FRONT_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_front_right_val[i] = aux_list[id]
            closest_cam_front_right_tokens_val[i] = aux_list_tokens[id]

            aux_list = self.sensors_val['images']['CAM_BACK']
            aux_list_tokens = self.sensors_val['tokens']['CAM_BACK']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_back_val[i] = aux_list[id]
            closest_cam_back_tokens_val[i] = aux_list_tokens[id]

            aux_list = self.sensors_val['images']['CAM_BACK_LEFT']
            aux_list_tokens = self.sensors_val['tokens']['CAM_BACK_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_back_left_val[i] = aux_list[id]
            closest_cam_back_left_tokens_val[i] = aux_list_tokens[id]

            aux_list = self.sensors_val['images']['CAM_BACK_RIGHT']
            aux_list_tokens = self.sensors_val['tokens']['CAM_BACK_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus_val[i,0])
            closest_cam_back_right_val[i] = aux_list[id]
            closest_cam_back_right_tokens_val[i] = aux_list_tokens[id]

        #Dictionaries
        self.closest_sensors_train = {
            'images': {
                'CAM_FRONT': closest_cam_front_train,
                'CAM_FRONT_LEFT': closest_cam_front_left_train,
                'CAM_FRONT_RIGHT': closest_cam_front_right_train,
                'CAM_BACK': closest_cam_back_train,
                'CAM_BACK_LEFT': closest_cam_back_left_train,
                'CAM_BACK_RIGHT': closest_cam_back_right_train
            },
            'tokens': {
                'CAM_FRONT': closest_cam_front_tokens_train,
                'CAM_FRONT_LEFT': closest_cam_front_left_tokens_train,
                'CAM_FRONT_RIGHT': closest_cam_front_right_tokens_train,
                'CAM_BACK': closest_cam_back_tokens_train,
                'CAM_BACK_LEFT': closest_cam_back_left_tokens_train,
                'CAM_BACK_RIGHT': closest_cam_back_right_tokens_train
            }
        }

        self.closest_sensors_val = {
            'images': {
                'CAM_FRONT': closest_cam_front_val,
                'CAM_FRONT_LEFT': closest_cam_front_left_val,
                'CAM_FRONT_RIGHT': closest_cam_front_right_val,
                'CAM_BACK': closest_cam_back_val,
                'CAM_BACK_LEFT': closest_cam_back_left_val,
                'CAM_BACK_RIGHT': closest_cam_back_right_val
            },
            'tokens': {
                'CAM_FRONT': closest_cam_front_tokens_val,
                'CAM_FRONT_LEFT': closest_cam_front_left_tokens_val,
                'CAM_FRONT_RIGHT': closest_cam_front_right_tokens_val,
                'CAM_BACK': closest_cam_back_tokens_val,
                'CAM_BACK_LEFT': closest_cam_back_left_tokens_val,
                'CAM_BACK_RIGHT': closest_cam_back_right_tokens_val
            }
        }


    def show_data(self, sensor = 'CAM_FRONT', labels = False, labels_array = None):

        aux_canbus_speed = self.can_bus['train']['speed']
        aux_canbus_steering = self.can_bus['train']['steering']
        aux_list = self.closest_sensors_train['images'][sensor]
        aux_list_tokens = self.closest_sensors_train['tokens'][sensor]
        for i in range(np.shape(aux_canbus_speed)[0]):

            print('Vehicle speed ' + str(aux_canbus_speed[i,1]))
            print('Vehicle steering ' + str(aux_canbus_steering[i,1]))

            if labels == True:
                print('Speed/Direction: ' + labels_array[i])

            cv.imshow(str(aux_list_tokens[i]), cv.imread(aux_list[i]))

            key = cv.waitKey(0)

            #ESCAPE key
            if key == 27:
                cv.destroyWindow(str(aux_list_tokens[i]))
                break
            #ENTER key
            elif key == 13:
                cv.destroyWindow(str(aux_list_tokens[i]))
                continue

    #Build labels
    #[i,0] straight, left, right.
    #[i,1] stop, accel, stoping.
    #We consider actual and previous data.
    def get_labels(self, type = 'train'):

        aux_canbus_speed = self.can_bus[type]['speed']
        aux_canbus_steering = self.can_bus[type]['steering']
        labels_array = np.empty(np.shape(aux_canbus_speed)[0], dtype = object)
        for i in range(np.shape(aux_canbus_speed)[0]):
            #First position does not have previous data
            if i == 0:
                diff_speed = aux_canbus_speed[i,1]
                diff_steering = aux_canbus_steering[i,1]

                if aux_canbus_steering[i,1] > 0:
                    #Left
                    steering = 1
                else:
                    #Right
                    steering = 0
            else:
                diff_speed = aux_canbus_speed[i,1] - aux_canbus_speed[i-1,1]
                diff_steering = aux_canbus_steering[i,1] - aux_canbus_steering[i-1,1]

                if aux_canbus_steering[i,1] > aux_canbus_steering[i-1,1]:
                    #Left
                    steering = 1
                else:
                    #Right
                    steering = 0

            if aux_canbus_speed[i,1] == 0:
                labels_1 = 'stop'
            elif diff_speed > 0:
                labels_1 = 'accel'
            else:
                labels_1 = 'stoping'

            if abs(diff_steering) > 0.6 and steering == 1:
                labels_2 = 'left'
            elif abs(diff_steering) > 0.6 and steering == 0:
                labels_2 = 'right'
            else:
                labels_2 = 'straight'

            labels_array[i] = labels_1 + '-' + labels_2

        return labels_array

    #Getters

    #CAN bus data
    def get_canbus_data(self, data = 'speed', type = 'train'):
        return self.can_bus[type][data]

    #Image arrays
    def get_sensors_data(self, sensor = 'CAM_FRONT', type = 'train'):
        if type == 'train':
            return self.sensors_train['images'][sensor], self.sensors_train['tokens'][sensor]

        elif type == 'val':
            return self.sensors_val['images'][sensor], self.sensors_val['tokens'][sensor]

    #Array containing the closest images to the can bus data
    def get_closest_sensors_data(self, sensor = 'CAM_FRONT', type = 'train'):
        if type == 'train':
            return self.closest_sensors_train['images'][sensor], self.closest_sensors_train['tokens'][sensor]

        elif type == 'val':
            return self.closest_sensors_val['images'][sensor], self.closest_sensors_val['tokens'][sensor]
