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
    def __init__(self, root_dir, len_hist=20, len_fut=30, use_yaw=False, random_rotation=False, normalize_angle=True):
        self.AFL = ArgoverseForecastingLoader(root_dir)
        self.len_hist = len_hist
        self.len_fut = len_fut
        self.use_yaw = use_yaw
        if self.use_yaw:
            self.feature_size = 3
        else:
            self.feature_size = 2
        self.random_rotation = random_rotation
        self.normalize_angle = normalize_angle
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
        trajectory = trajectory - trajectory[self.len_hist-1]
        if self.use_yaw:
            traj_diff = trajectory[1:] - trajectory[:-1]
            yaw = np.arctan2(traj_diff[:, 1:2], traj_diff[:, 0:1])
            yaw = np.concatenate((yaw[0:1, :], yaw), axis=0)
            trajectory = np.concatenate((trajectory, yaw), axis=1)
        if self.random_rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            trajectory[:, :2] = self.scene_rotation(trajectory[:, :2], angle)
            if self.use_yaw:
                trajectory[:, 2] += angle
                trajectory[:, 2] = (trajectory[:, 2] + np.pi) % (2*np.pi) - np.pi
        elif self.normalize_angle:
            traj_diff = trajectory[self.len_hist-1] - trajectory[0]
            angle = -np.arctan2(traj_diff[1:2], traj_diff[0:1])
            trajectory[:, :2] = self.scene_rotation(trajectory[:, :2], angle)
            if self.use_yaw:
                trajectory[:, 2] += angle

        return trajectory

    ## Collate function for dataloader
    def collate_fn(self, samples):
        batch_size = len(samples)
        # Initialize history, future
        hist_batch = np.zeros([self.len_hist, batch_size, self.feature_size])
        fut_batch = np.zeros([self.len_fut, batch_size, self.feature_size])

        for sampleId, trajectory in enumerate(samples):
            hist_batch[:, sampleId, :] = trajectory[:self.len_hist, :]
            fut_batch[:, sampleId, :] = trajectory[self.len_hist:, :]

        hist_batch = torch.from_numpy(hist_batch.astype('float32'))
        fut_batch = torch.from_numpy(fut_batch.astype('float32'))
        return hist_batch, fut_batch

