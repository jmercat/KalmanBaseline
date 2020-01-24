from fusion_indices import *
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

## Indices of the global information in the csv files
GLOBAL_TIME = 0
EGO_OFFSET = 1
N_OBJ = 7

## Indices of the ego information in the csv files
EGO_ID = 0
EGO_DX = 1
EGO_DY = 2
EGO_DYAW = 3
EGO_VX = 4
EGO_VY = 5
EGO_DATA_LEN = 6


## After global and ego information, n_obj times this amount of info about neighbor objects is available
OBJ_OFFSET = 8
OBJ_ID = 0
OBJ_X = 1
OBJ_Y = 2
OBJ_YAW = 3
OBJ_WIDTH = 4
OBJ_LENGTH = 5
OBJ_VX = 6
OBJ_VY = 7
OBJ_CLASS = 8
OBJ_DATA_LEN = 9

## After object information, the numbner of lines n_line is written then n_lines times this info is available
LINE_ID = 0
LINE_X = 1
LINE_Y = 2
LINE_THETA = 3
LINE_C = 4
LINE_DC = 5
LINE_LENGTH = 6
LINE_DATA_LEN = 7

class FusionDataset(Dataset):
    def __init__(self, dataset_path, random_rotation=False, random_translation=False, use_yaw=False):
        super(FusionDataset, self).__init__()
        self.load_dataset(dataset_path)
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.translation_distance_std = 2
        self.use_yaw = use_yaw
        if use_yaw:
            self.data_to_get = [OBJ_X, OBJ_Y, OBJ_YAW]
        else:
            self.data_to_get = [OBJ_X, OBJ_Y]
        self.hist_len = int(1000*2/40)
        self.fut_len = int(1000*3/40)
        self.time_len = self.hist_len + self.fut_len
        self.min_num_obs = 10
        self.down_sampling = 1

    def load_dataset(self, path):
        with open(path, 'rb') as handle:
            self.dataset = pickle.load(handle)

    def __len__(self):
        return len(self.dataset['traj'])

    @staticmethod
    def scene_rotation(coor, angle):
        rot_matrix = np.zeros((2, 2))
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix[0, 0] = c
        rot_matrix[0, 1] = -s
        rot_matrix[1, 0] = s
        rot_matrix[1, 1] = c
        coor = np.matmul(rot_matrix, np.expand_dims(coor, axis=-1))
        return coor.squeeze(-1)

    def __getitem__(self, item):
        traj = self.dataset['traj'][item][:, 0:1, self.data_to_get]

        if self.random_translation:
            distance = np.random.normal([0, 0], self.translation_distance_std, 2)
            traj[:, :, :2] = traj[:, :, :2] + distance

        if self.random_rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            traj[:, :, :2] = self.scene_rotation(traj[:, :, :2], angle)
            if self.use_yaw:
                traj[:, :, 2] += angle
        return traj[:, 0, :]

    def collate_fn(self, samples):
        traj = samples[0]
        time_len = traj.shape[0]
        assert time_len == self.time_len
        time_len = self.time_len // self.down_sampling
        hist_len = self.hist_len // self.down_sampling
        data_batch = np.zeros([time_len, len(samples), len(self.data_to_get)])

        for i, traj in enumerate(samples):
            data_batch[:, i, :] = traj[:self.down_sampling*time_len:self.down_sampling, :]

        data_batch = torch.from_numpy(data_batch.astype('float32'))

        return data_batch[:hist_len], data_batch[hist_len:]
