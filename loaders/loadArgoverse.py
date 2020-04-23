from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import os
import sys

# Argoverse is imported this way because it could not be installed properly on windows
current_path = os.getcwd()
sys.path.insert(1, current_path+'/../Argoverse')
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


class ArgoverseDataset(Dataset):
    def __init__(self, root_dir, args):
        self.AFL = ArgoverseForecastingLoader(root_dir)
        self.use_yaw = args.use_yaw
        self.hist_len = int(args.time_hist / args.dt)
        self.fut_len = int(args.time_pred / args.dt)
        self.time_len = self.hist_len + self.fut_len
        if self.use_yaw:
            self.feature_size = 3
        else:
            self.feature_size = 2
        self.random_rotation = args.random_rotation
        self.normalize_angle = args.normalize_angle
        self.down_sampling = args.down_sampling
    def __len__(self):
        return len(self.AFL)

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

    def __getitem__(self, idx):
        trajectory = self.AFL[idx]
        trajectory = trajectory.agent_traj
        trajectory = trajectory - trajectory[self.hist_len - 1]

        if self.random_rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            trajectory[:, :2] = self.scene_rotation(trajectory[:, :2], angle)
            # if self.use_yaw:
            #     trajectory[:, 2] += angle
            #     trajectory[:, 2] = (trajectory[:, 2] + np.pi) % (2*np.pi) - np.pi
        elif self.normalize_angle:
            traj_diff = trajectory[self.hist_len - 1] - trajectory[0]
            angle = -np.arctan2(traj_diff[1:2], traj_diff[0:1])
            trajectory[:, :2] = self.scene_rotation(trajectory[:, :2], angle)
            # if self.use_yaw:
            #     trajectory[:, 2] += angle
        if self.use_yaw:
            traj_diff = trajectory[1:] - trajectory[:-1]
            yaw = np.arctan2(traj_diff[:, 1:2], traj_diff[:, 0:1])
            yaw = (yaw[1:] + yaw[:-1])/2
            yaw = np.concatenate((yaw[0:2, :], yaw), axis=0)
            trajectory = np.concatenate((trajectory, yaw), axis=1)
        return trajectory

    ## Collate function for dataloader
    def collate_fn(self, samples):
        batch_size = len(samples)

        hist_len = self.hist_len
        fut_len = self.fut_len
        time_len = hist_len + fut_len

        # Initialize history, future
        hist_batch = np.zeros([hist_len, batch_size, self.feature_size])
        fut_batch = np.zeros([fut_len, batch_size, self.feature_size])

        for sampleId, trajectory in enumerate(samples):
            hist_batch[:, sampleId, :] = trajectory[:hist_len*self.down_sampling:self.down_sampling, :]
            fut_batch[:, sampleId, :] = \
                trajectory[hist_len*self.down_sampling:time_len*self.down_sampling:self.down_sampling, :]

        hist_batch = torch.from_numpy(hist_batch.astype('float32'))
        fut_batch = torch.from_numpy(fut_batch.astype('float32'))
        return hist_batch, fut_batch

