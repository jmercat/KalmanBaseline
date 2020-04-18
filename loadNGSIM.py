from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py

### This is a modified version of the code : https://github.com/nachiket92/conv-social-pooling

class NGSIMDataset(Dataset):

    def __init__(self, mat_traj_file, mat_tracks_file, args):
        """
        :param mat_traj_file: name of trajectory file generated from the matlab preprocessing
        :param mat_tracks_file: name of track file generated from the matlab preprocessing
        """
        self.D = np.array(h5py.File(mat_traj_file, mode='r')['traj']).transpose()
        self.T = scp.loadmat(mat_tracks_file)['tracks']
        self.use_yaw = args.use_yaw
        self.hist_len = int(args.time_hist / args.dt)
        self.fut_len = int(args.time_pred / args.dt)
        self.time_len = self.hist_len + self.fut_len
        self.down_sampling = args.down_sampling  # down sampling rate of all sequences
        self.feature_size = 2 + int(args.use_yaw)
        self.random_rotation = args.random_rotation
        self.normalize_angle = args.normalize_angle
        self.unit_conversion = args.unit_conversion

    def __len__(self):
        return len(self.D)

    def _add_yaw(self, trajectory):
        traj_diff = trajectory[1:] - trajectory[:-1]
        yaw = np.arctan2(traj_diff[:, 1:2], traj_diff[:, 0:1])
        yaw = np.concatenate((yaw[0:1, :], yaw), axis=0)
        return np.concatenate((trajectory, yaw), axis=1)

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

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]

        fut = self.getFuture(vehId, t, dsId)*self.unit_conversion
        if fut.shape[0] < 5:
            return None, None
        hist = self.getHistory(vehId, t, vehId, dsId)*self.unit_conversion
        if hist.shape[0] < 5:
            return None, None
        if self.use_yaw:
            hist = self._add_yaw(hist)
        if self.random_rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            hist[:, :2] = self.scene_rotation(hist[:, :2], angle)
            fut[:, :2] = self.scene_rotation(fut[:, :2], angle)
            if self.use_yaw:
                hist[:, 2] += angle
                hist[:, 2] = (hist[:, 2] + np.pi) % (2*np.pi) - np.pi
        elif self.normalize_angle:
            traj_diff = hist[-1] - hist[0]
            angle = -np.arctan2(traj_diff[1:2], traj_diff[0:1])
            hist[:, :2] = self.scene_rotation(hist[:, :2], angle)
            fut[:, :2] = self.scene_rotation(fut[:, :2], angle)
            if self.use_yaw:
                hist[:, 2] += angle

        return hist, fut


    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - (self.hist_len - 1)*self.down_sampling)
                enpt = np.argwhere(vehTrack[:, 0] == t).item()+self.down_sampling
                hist = vehTrack[stpt:enpt:self.down_sampling, 1:3] - refPos

            if len(hist) < self.hist_len:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item()+self.down_sampling
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + (self.fut_len + 1)*self.down_sampling)
        fut = vehTrack[stpt:enpt:self.down_sampling, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        maxlen = self.hist_len

        # Initialize history, future
        time_size = self.fut_len
        hist_batch = torch.zeros(maxlen, len(samples), self.feature_size)
        fut_batch = torch.zeros(time_size, len(samples), 2)

        for sampleId, (hist, fut) in enumerate(samples):
            if hist is not None:
                # Set up history, future, lateral maneuver and longitudinal maneuver batches:
                hist_batch[-len(hist):, sampleId, :] = torch.from_numpy(hist)
                fut_batch[:len(fut), sampleId, :] = torch.from_numpy(fut)

        return hist_batch, fut_batch

