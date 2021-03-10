#Sweeps frame rate: 10Hz

import numpy as np
import cv2 as cv
import pandas as pd

#Pytorch
import torch
from torch.utils.data import Dataset

#NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

#Progress bar
from tqdm import tqdm

class DataLoaderReg(Dataset):

    def __init__(self, HOME_ROUTE = '/data/sets/nuscenes/', mode = 'train', canbus_scenes = 1111, sensor_scenes = 850, transform = None):
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

        #Returns to which scene the sample belongs
        def get_which_scene(id, samples):
            aux = 0
            for i in range(len(samples)):
                aux += samples[i]
                if id < aux:
                    scene = i
                    break

            return scene


        message_name = 'vehicle_monitor'
        if mode == 'train':
            scene_name = 'scene-0001'
            vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
            vehicle_speed = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])
            ir = 2
        else:
            scene_name = 'scene-0003'
            vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
            vehicle_speed = np.array([(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux])
            ir = 4

        for i in range(ir, canbus_scenes):
            type, scene_name = get_listed_scene(i)
            if type == mode:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_speed = np.append(vehicle_speed, [(m['utime'], m['vehicle_speed']) for m in vehicle_monitor_aux], axis = 0)
        if mode == 'train':
            scene_name = 'scene-0001'
            vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
            vehicle_steering = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])
            ir = 2
        else:
            scene_name = 'scene-0003'
            vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
            vehicle_steering = np.array([(m['utime'], m['steering']) for m in vehicle_monitor_aux])
            ir = 4

        for i in range(ir, canbus_scenes):
            type, scene_name = get_listed_scene(i)
            if type == mode:
                vehicle_monitor_aux = nusc_can.get_messages(scene_name, message_name)
                vehicle_steering = np.append(vehicle_steering, [(m['utime'], m['steering']) for m in vehicle_monitor_aux], axis = 0)

        #Dictionaries
        self.can_bus = {
            'speed': vehicle_speed,
            'steering': vehicle_steering
        }


        # Sequence lenght
        img_scene = []
        nbr_scenes = 0
        nbr_images = 0

        #Load images
        #CAM_FRONT
        sensor = 'CAM_FRONT'

        cam_front = []
        cam_front_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]
            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                # Number of scenes
                nbr_scenes += 1

                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_data['next'] == "":
                    token = cam_front_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_data['filename'])

                    cam_front.append(route)
                    cam_front_tokens.append(token)

                    cam_front_data = nusc.get('sample_data', cam_front_data['next'])

                    nbr_images += 1

                # Images per scene
                img_scene.append(nbr_images)
                nbr_images = 0

        #Converts the input in an array
        cam_front = np.asarray(cam_front)
        cam_front_tokens = np.asarray(cam_front_tokens)

        #CAM_FRONT_LEFT
        sensor = 'CAM_FRONT_LEFT'

        cam_front_left = []
        cam_front_left_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_left_data['next'] == "":
                    token = cam_front_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_left_data['filename'])

                    cam_front_left.append(route)
                    cam_front_left_tokens.append(token)

                    cam_front_left_data = nusc.get('sample_data', cam_front_left_data['next'])

        #Converts the input in an array
        cam_front_left = np.asarray(cam_front_left)
        cam_front_left_tokens = np.asarray(cam_front_left_tokens)

        #CAM_FRONT_RIGHT
        sensor = 'CAM_FRONT_RIGHT'

        cam_front_right = []
        cam_front_right_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_front_right_data['next'] == "":
                    token = cam_front_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_front_right_data['filename'])

                    cam_front_right.append(route)
                    cam_front_right_tokens.append(token)

                    cam_front_right_data = nusc.get('sample_data', cam_front_right_data['next'])

        #Converts the input in an array
        cam_front_right = np.asarray(cam_front_right)
        cam_front_right_tokens = np.asarray(cam_front_right_tokens)

        #CAM_BACK
        sensor = 'CAM_BACK'

        cam_back = []
        cam_back_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_data['next'] == "":
                    token = cam_back_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_data['filename'])

                    cam_back.append(route)
                    cam_back_tokens.append(token)

                    cam_back_data = nusc.get('sample_data', cam_back_data['next'])

        #Converts the input in an array
        cam_back = np.asarray(cam_back)
        cam_back_tokens = np.asarray(cam_back_tokens)


        #CAM_BACK_LEFT
        sensor = 'CAM_BACK_LEFT'

        cam_back_left = []
        cam_back_left_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_left_data['next'] == "":
                    token = cam_back_left_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_left_data['filename'])

                    cam_back_left.append(route)
                    cam_back_left_tokens.append(token)

                    cam_back_left_data = nusc.get('sample_data', cam_back_left_data['next'])


        #Converts the input in an array
        cam_back_left = np.asarray(cam_back_left)
        cam_back_left_tokens = np.asarray(cam_back_left_tokens)

        #CAM_BACK_RIGHT
        sensor = 'CAM_BACK_RIGHT'

        cam_back_right = []
        cam_back_right_tokens = []
        for i in range(sensor_scenes):
            my_scene = nusc.scene[i]

            type, scene = get_listed_scene(-1, my_scene['name'])

            if type == mode:
                my_sample = nusc.get('sample', my_scene['first_sample_token'])

                cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
                while not cam_back_right_data['next'] == "":
                    token = cam_back_right_data['timestamp']
                    route = HOME_ROUTE + str(cam_back_right_data['filename'])

                    cam_back_right.append(route)
                    cam_back_right_tokens.append(token)

                    cam_back_right_data = nusc.get('sample_data', cam_back_right_data['next'])


        #Converts the input in an array
        cam_back_right = np.asarray(cam_back_right)
        cam_back_right_tokens = np.asarray(cam_back_right_tokens)

        #Dictionaries
        self.sensors_data = {
            'images': {
                'CAM_FRONT': cam_front,
                'CAM_FRONT_LEFT': cam_front_left,
                'CAM_FRONT_RIGHT': cam_front_right,
                'CAM_BACK': cam_back,
                'CAM_BACK_LEFT': cam_back_left,
                'CAM_BACK_RIGHT': cam_back_right
            },
            'tokens':{
                'CAM_FRONT': cam_front_tokens,
                'CAM_FRONT_LEFT': cam_front_left_tokens,
                'CAM_FRONT_RIGHT': cam_front_right_tokens,
                'CAM_BACK': cam_back_tokens,
                'CAM_BACK_LEFT': cam_back_left_tokens,
                'CAM_BACK_RIGHT': cam_back_right_tokens
            }
        }

        #Build the closest arrays
        aux_canbus = self.can_bus['speed']
        # Sequence length
        self.seq_length = np.zeros(nbr_scenes, dtype=np.int64)

        closest_cam_front = np.empty(np.shape(aux_canbus)[0], dtype=np.dtype('<U122'))
        closest_cam_front_tokens = np.empty(np.shape(aux_canbus)[0], dtype=np.int64)

        closest_cam_front_left = np.empty(np.shape(aux_canbus)[0], dtype=np.dtype('<U122'))
        closest_cam_front_left_tokens = np.empty(np.shape(aux_canbus), dtype=np.int64)

        closest_cam_front_right = np.empty(np.shape(aux_canbus)[0], dtype=np.dtype('<U122'))
        closest_cam_front_right_tokens = np.empty(np.shape(aux_canbus)[0], dtype=np.int64)

        closest_cam_back = np.empty(np.shape(aux_canbus)[0], dtype=np.dtype('<U122'))
        closest_cam_back_tokens = np.empty(np.shape(aux_canbus)[0], dtype=np.int64)

        closest_cam_back_left = np.empty(np.shape(aux_canbus), dtype=np.dtype('<U122'))
        closest_cam_back_left_tokens = np.empty(np.shape(aux_canbus)[0], dtype=np.int64)

        closest_cam_back_right = np.empty(np.shape(aux_canbus)[0], dtype=np.dtype('<U122'))
        closest_cam_back_right_tokens = np.empty(np.shape(aux_canbus)[0], dtype=np.int64)


        print('Building closest arrays')
        for i in tqdm(range(np.shape(aux_canbus)[0])):

            #CAM FRONT
            aux_list = self.sensors_data['images']['CAM_FRONT']
            aux_list_tokens = self.sensors_data['tokens']['CAM_FRONT']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_front[i] = aux_list[id]
            closest_cam_front_tokens[i] = aux_list_tokens[id]

            #Sequence length
            s = get_which_scene(id, img_scene)
            self.seq_length[s] += 1

            #CAM FRONT LEFT
            aux_list = self.sensors_data['images']['CAM_FRONT_LEFT']
            aux_list_tokens = self.sensors_data['tokens']['CAM_FRONT_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_front_left[i] = aux_list[id]
            closest_cam_front_left_tokens[i] = aux_list_tokens[id]

            #CAM FRONT RIGHT
            aux_list = self.sensors_data['images']['CAM_FRONT_RIGHT']
            aux_list_tokens = self.sensors_data['tokens']['CAM_FRONT_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_front_right[i] = aux_list[id]
            closest_cam_front_right_tokens[i] = aux_list_tokens[id]

            #CAM BACK
            aux_list = self.sensors_data['images']['CAM_BACK']
            aux_list_tokens = self.sensors_data['tokens']['CAM_BACK']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_back[i] = aux_list[id]
            closest_cam_back_tokens[i] = aux_list_tokens[id]

            #CAM BACK LEFT
            aux_list = self.sensors_data['images']['CAM_BACK_LEFT']
            aux_list_tokens = self.sensors_data['tokens']['CAM_BACK_LEFT']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_back_left[i] = aux_list[id]
            closest_cam_back_left_tokens[i] = aux_list_tokens[id]

            #CAM BACK RIGHT
            aux_list = self.sensors_data['images']['CAM_BACK_RIGHT']
            aux_list_tokens = self.sensors_data['tokens']['CAM_BACK_RIGHT']
            id = get_closest(aux_list_tokens, aux_canbus[i,0])
            closest_cam_back_right[i] = aux_list[id]
            closest_cam_back_right_tokens[i] = aux_list_tokens[id]

        #Dictionaries
        self.sensors_labelled_data = {
            'images': {
                'CAM_FRONT': closest_cam_front,
                'CAM_FRONT_LEFT': closest_cam_front_left,
                'CAM_FRONT_RIGHT': closest_cam_front_right,
                'CAM_BACK': closest_cam_back,
                'CAM_BACK_LEFT': closest_cam_back_left,
                'CAM_BACK_RIGHT': closest_cam_back_right
            },
            'tokens': {
                'CAM_FRONT': closest_cam_front_tokens,
                'CAM_FRONT_LEFT': closest_cam_front_left_tokens,
                'CAM_FRONT_RIGHT': closest_cam_front_right_tokens,
                'CAM_BACK': closest_cam_back_tokens,
                'CAM_BACK_LEFT': closest_cam_back_left_tokens,
                'CAM_BACK_RIGHT': closest_cam_back_right_tokens
            }
        }

        self.transform = transform


    def __len__(self):
        return np.shape(self.seq_length)[0]

    def true_length(self):
        aux = 0
        for i in range(np.shape(self.seq_length)[0]):
            aux += self.seq_length[i]

        return aux

    def __getitem__(self, idx):
        seq = self.seq_length[idx]

        id_s = 0
        id = []
        images = []
        ndata = []
        for i in range(idx):
            id_s += self.seq_length[i]

        if id_s > 0:
            id_s = id_s - 1

        for i in range(seq):
            path = self.sensors_labelled_data['images']['CAM_FRONT'][id_s+i]
            image = cv.imread(path)
            image = image / 255
            image = image.astype(np.float32)

            images.append(image)
            nd = np.array([self.can_bus['speed'][id_s+i,1], self.can_bus['steering'][id_s+i,1]])
            nd = nd.astype(np.float32)
            ndata.append(nd)

        train = {'image': np.array(images), 'can_bus': ndata}

        if self.transform:
            train = self.transform(train)

        return train


    def show_data(self, sensor = 'CAM_FRONT'):

        aux_canbus_speed = self.can_bus['speed']
        aux_canbus_steering = self.can_bus['steering']
        aux_list = self.sensors_labelled_data['images'][sensor]
        aux_list_tokens = self.sensors_labelled_data['tokens'][sensor]
        for i in range(np.shape(aux_canbus_speed)[0]):

            print('Vehicle speed ' + str(aux_canbus_speed[i,1]))
            print('Vehicle steering ' + str(aux_canbus_steering[i,1]))

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


    def get_seq_length(self):
        return self.seq_length
