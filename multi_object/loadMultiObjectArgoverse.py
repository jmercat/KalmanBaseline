from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd
from numba import jit

import pickle
import os, os.path


class MultiObjectArgoverseDataset(Dataset):
    def __init__(self, dataset_dir, random_rotation=False, random_translation=False, normalize=False, scale_factor=1, limit_file_number=None, get_id=False):
        super(MultiObjectArgoverseDataset, self).__init__()
        self.get_id = get_id
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.n_points_slope = 5
        self.load_dataset(dataset_dir, limit_file_number)
        self.idx_list = list(self.dataset.keys())
        self.down_sampling = 1
        self.time_len = 50
        self.hist_len = 20
        self.dist_max = 30
        self.min_num_obs = 4
        self.max_num_lanes = 30
        self.max_size_lane = 10
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.angle_std = np.pi/8
        self.translation_distance_std = 0.5
        # self.city_lanes = {}

    def load_dataset(self, dataset_dir, limit_file_number):
        num_files = len([name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))])
        if limit_file_number is not None:
            num_files_max = min(limit_file_number, num_files)
        else:
            num_files_max = num_files
        # file_inex_list = np.random.randint(0, num_files - 1, num_files_max)
        print('Loading dataset from %d files.' % num_files_max)
        # traj = []
        # mask_traj = []
        # lanes = []
        # mask_lanes = []
        # id = []
        # for i in file_inex_list:
        #     with open(dataset_dir + str(i) + '.pickle', 'rb') as handle:
        #         data_temp = pickle.load(handle)
        #         traj = traj + [data_temp['traj'][i] / self.scale_factor for i in range(len(data_temp['traj']))]
        #         mask_traj = mask_traj + data_temp['mask_traj']
        #         lanes = lanes + [data_temp['lanes'][i] / self.scale_factor for i in range(len(data_temp['lanes']))]
        #         mask_lanes = mask_lanes + data_temp['mask_lanes']
        #         if self.get_id:
        #             id = id + data_temp['seq_id']
        with open(dataset_dir + 'data+lanes.pickle', 'rb') as handle:
            self.dataset = pickle.load(handle)
            # data_temp = pickle.load(handle)
            # traj = traj + [data_temp['traj'][i] / self.scale_factor for i in range(len(data_temp['traj']))]
            # mask_traj = mask_traj + data_temp['mask_traj']
            # lanes = lanes + [data_temp['lanes'][i] / self.scale_factor for i in range(len(data_temp['lanes']))]
            # mask_lanes = mask_lanes + data_temp['mask_lanes']
            # if self.get_id:
            #     id = id + data_temp['seq_id']
        # if self.get_id:
        #     return {'traj': traj, 'mask_traj': mask_traj,
        #            'lanes': lanes, 'mask_lanes': mask_lanes, 'seq_id': id}
        # else:
        #     return {'traj': traj, 'mask_traj': mask_traj,
        #            'lanes': lanes, 'mask_lanes': mask_lanes}


    def __len__(self):
        return len(self.dataset)

    def _get_lane_by_track_id(self, df, track_id):
        return df[df["TRACK_ID"] == track_id]

    def __getitem__(self, idx: int):
        idx = self.idx_list[idx]
        rel_coor_all = self.dataset[idx]['traj']
        mask_all = self.dataset[idx]['mask_traj']
        rel_lane_all = self.dataset[idx]['lanes']
        mask_lane_all = self.dataset[idx]['mask_lanes']
        mean_pos = self.dataset[idx]['mean_pos']

        if self.normalize:
            X = rel_coor_all[self.hist_len - self.n_points_slope:self.hist_len, 0, 0]
            Y = rel_coor_all[self.hist_len - self.n_points_slope:self.hist_len, 0, 1]
            mdX = np.mean(X[1:] - X[:-1])
            mdY = np.mean(Y[1:] - Y[:-1])
            angle = -np.arctan2(mdY, mdX)
            angle += np.pi / 4

            if self.random_rotation:
                if self.normalize:
                    angle = angle + np.random.normal(0, self.angle_std)
                else:
                    angle = angle + np.random.uniform(-np.pi, np.pi)
            if self.random_translation:
                distance = np.random.normal([0, 0], self.translation_distance_std, 2)
                rel_coor_all = rel_coor_all + mask_all[:, :, None]*distance
                rel_lane_all = rel_lane_all + mask_lane_all[:, :, None]*distance

            rel_coor_all = self.scene_rotation(rel_coor_all, angle)
            rel_lane_all = self.scene_rotation(rel_lane_all, angle)

            return rel_coor_all, mask_all, rel_lane_all, mask_lane_all, angle, mean_pos
        else:
            if self.random_translation:
                distance = np.random.normal([0, 0], self.translation_distance_std, 2)
                rel_coor_all = rel_coor_all + mask_all[:, :, None]*distance
                rel_lane_all = rel_lane_all + mask_lane_all[:, :, None]*distance

            if self.random_rotation:
                angle = np.random.uniform(0, 2*np.pi)
                rel_coor_all = self.scene_rotation(rel_coor_all, angle)
                rel_lane_all = self.scene_rotation(rel_lane_all, angle)
            else:
                angle = 0

            return rel_coor_all, mask_all, rel_lane_all, mask_lane_all, angle, mean_pos

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

    @staticmethod
    def _count_last_obs(coor, hist_len=None):
        if hist_len is not None:
            time_len_coor = np.sum(np.sum(np.cumprod((coor[hist_len-1::-1] != 0), 0), 2) > 0, 0)
        else:
            time_len_coor = np.sum(np.sum(np.cumprod((coor[coor.shape[0]-1::-1] != 0), 0), 2) > 0, 0)
        return time_len_coor

    def collate_fn(self, samples):
        time_len = self.time_len // self.down_sampling
        hist_len = self.hist_len // self.down_sampling

        max_n_vehicle = 0
        max_n_lanes = 0
        for coor, mask, lanes, mask_lanes, angle, mean_pos in samples:
            # time_len_coor = self._count_last_obs(coor, hist_len*self.down_sampling)
            # num_vehicle = np.sum(time_len_coor > self.min_num_obs)
            num_vehicle = coor.shape[1]
            num_lanes = len(lanes)
            if num_lanes > 0:
                points_len = lanes.shape[1]
            max_n_vehicle = max(num_vehicle, max_n_vehicle)
            max_n_lanes = max(num_lanes, max_n_lanes)
        if max_n_vehicle <= 0:
            raise RuntimeError
        data_batch = np.zeros([time_len, len(samples), max_n_vehicle, 2])
        mask_batch = np.zeros([time_len, len(samples), max_n_vehicle])
        lane_batch = np.zeros([self.max_size_lane, len(samples), self.max_num_lanes, 2])
        mask_lane_batch = np.zeros([self.max_size_lane, len(samples), self.max_num_lanes])
        angle_batch = np.zeros([len(samples)])
        mean_pos_batch = np.zeros([len(samples), 2])

        for sample_ind, (coor, mask, lanes, mask_lanes, angle, mean_pos) in enumerate(samples):
            # args = np.argwhere(self._count_last_obs(coor, hist_len*self.down_sampling) > self.min_num_obs)[:, 0]
            data_batch[:, sample_ind, :coor.shape[1], :] = coor[::self.down_sampling, :, :]
            mask_batch[:, sample_ind, :mask.shape[1]] = mask[::self.down_sampling, :]
            lane_batch[:, sample_ind, :, :] = lanes
            mask_lane_batch[:, sample_ind, :] = mask_lanes
            angle_batch[sample_ind] = angle
            mean_pos_batch[sample_ind, :] = mean_pos

        mask_past = mask_batch[:hist_len]

        mask_fut = np.cumprod(mask_batch[hist_len-self.min_num_obs:], 0)[self.min_num_obs:]
        # mask_past = np.cumprod(mask_past[::-1], 0)[::-1] #should not change anything

        data_batch = torch.from_numpy(data_batch.astype('float32'))
        mask_past = torch.from_numpy(mask_past.astype('bool'))
        mask_fut = torch.from_numpy(mask_fut.astype('bool'))
        lane_batch = torch.from_numpy(lane_batch.astype('float32'))
        mask_lane_batch = torch.from_numpy(mask_lane_batch.astype('bool'))

        fut = data_batch[hist_len:]
        angle = fut - data_batch[hist_len-1:-1]
        angle = torch.atan2(fut[:, :, :, 1], fut[:, :, :, 0])*mask_fut
        fut = torch.cat((fut, angle.unsqueeze(-1)), dim=-1)

        return data_batch[:hist_len], fut, mask_past, mask_fut, lane_batch, mask_lane_batch


