import torch
import numpy as np
import h5py
import scipy.io as scp
from torch.utils.data import Dataset

## Helper function for log sum exp calculation:
def logsumexp(inputs, mask=None, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    if mask is None:
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    else:
        # mask_veh = mask.unsqueeze(3)
        inputs_s = (inputs - s)#.masked_fill(mask_veh == 0, -1e3)
        outputs = s + inputs_s.exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

class MultiObjectNGSIMDataset(Dataset):

    def __init__(self, mat_traj_file, mat_tracks_file, args):
        self.D = np.array(h5py.File(mat_traj_file, 'r')['traj']).transpose()
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

    ## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
    def maskedNLLTest(self, fut_pred, proba_man, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True,
                      avg_along_time=False):
        if use_maneuvers:
            if torch.cuda.is_available():
                acc = torch.zeros(len(proba_man), op_mask.shape[0], op_mask.shape[1]).cuda()
            else:
                acc = torch.zeros(len(proba_man), op_mask.shape[0], op_mask.shape[1])
            for k in range(len(proba_man)):
                # for l in range(num_lat_classes):
                # wts = lat_pred[:, :, l] * lon_pred[:, :, k]
                wts = proba_man[k]
                # wts = wts.repeat(len(fut_pred[0]), 1)
                # wts = wts.view(-1, wts.shape[1])
                y_pred = fut_pred[k]
                y_gt = fut
                muX = y_pred.narrow(2, 0, 1)
                muY = y_pred.narrow(2, 1, 1)
                sigX = y_pred.narrow(2, 2, 1) + 1e-6
                sigY = y_pred.narrow(2, 3, 1) + 1e-6
                rho = y_pred.narrow(2, 4, 1)
                ohr = 1 / (1 - rho * rho + 1e-6)
                x = y_gt.narrow(2, 0, 1)
                y = y_gt.narrow(2, 1, 1)
                diff_x = (x - muX)*self.std[0, 0]
                diff_y = (y - muY)*self.std[1, 0]
                sigX = sigX * self.std[0, 0]
                sigY = sigY * self.std[1, 0]
                out = -ohr * (diff_x * diff_x / (sigX * sigX) + diff_y * diff_y / (sigY * sigY) -
                              2 * rho * diff_x * diff_y / (sigX * sigY)) - \
                      torch.log(sigX * sigX * sigY * sigY * (1 - rho * rho) + 1e-6)
                acc[k, :, :] = (0.5 * out.squeeze() + torch.log(wts + 1e-6))

            acc = -logsumexp(acc, dim=0)
            submask = op_mask.narrow(2, 0, 1).squeeze()
            acc = acc * submask
            if avg_along_time:
                lossVal = torch.sum(acc) / torch.sum(submask)
                return lossVal
            else:
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(submask, dim=1)
                return lossVal, counts
        else:
            if torch.cuda.is_available():
                acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
            else:
                acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred.narrow(2, 0, 1)
            muY = y_pred.narrow(2, 1, 1)
            sigX = y_pred.narrow(2, 2, 1)
            sigY = y_pred.narrow(2, 3, 1)
            rho = y_pred.narrow(2, 4, 1)
            ohr = 1 / (1 - rho * rho)
            x = y_gt.narrow(2, 0, 1)
            y = y_gt.narrow(2, 1, 1)
            diff_x = (x - muX)
            diff_y = (y - muY)
            out = ohr * (diff_x * diff_x / (sigX * sigX) + diff_y * diff_y / (sigY * sigY) -
                         2 * rho * diff_x * diff_y / (sigX * sigY)) + \
                  torch.log(sigX * sigX * sigY * sigY * (1 - rho * rho))
            acc[:, :, 0] = 0.5 * out
            submask = op_mask.narrow(2, 0, 1)
            acc = acc * submask
            if avg_along_time:
                lossVal = torch.sum(acc[:, :, 0]) / torch.sum(submask)
                return lossVal
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(submask, dim=1)
                return lossVal, counts

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 10:]
        neighbors = []
        neighbors_fut = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        fut = self.getFuture(vehId, t, dsId)*self.unit_conversion
        if fut.shape[0] < 5:
            return None, None, None, None
        hist = self.getHistory(vehId, t, vehId, dsId)*self.unit_conversion
        if hist.shape[0] < 5:
            return None, None, None, None
        if self.use_yaw:
            hist = self._add_yaw(hist)
        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            hist_nbr = self.getHistory(i.astype(int), t, vehId, dsId)*self.unit_conversion
            if self.use_yaw:
                hist_nbr = self._add_yaw(hist_nbr)
            neighbors.append(hist_nbr)
            neighbors_fut.append(self.getFuture2(i.astype(int), t, vehId, dsId)*self.unit_conversion)

        return hist, fut, neighbors, neighbors_fut

    def mean_smooth_derivative(self, x):
        xp = np.empty(x.shape[0])
        xp[1:] = x[1:] - x[:-1]
        xp[0] = (xp[1] + xp[2]) / 2
        xp[1:-1] = (xp[0:-2] + xp[1:-1] + xp[2:]) / 3
        xp[-1] = (2 * xp[-2] + xp[-1]) / 3
        return xp

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
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + self.down_sampling
                hist = vehTrack[stpt:enpt:self.down_sampling, 1:3] - refPos

            if len(hist) < 5:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.down_sampling
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + (self.fut_len + 1)*self.down_sampling)
        fut = vehTrack[stpt:enpt:self.down_sampling, 1:3] - refPos
        return fut

    ## Helper function to get track future
    def getFuture2(self, vehId, t, refVehId, dsId):
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
                stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.down_sampling
                enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + (self.fut_len + 1)*self.down_sampling)
                fut = vehTrack[stpt:enpt:self.down_sampling, 1:3] - refPos
            if len(fut) < 5:
                return np.empty([0, 2])
            return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        maxlen = self.hist_len
        for _, _, nbrs, _ in samples:
            if nbrs is not None:
                nbr_batch_size = max(sum([len(nbrs[i]) != 0 for i in range(len(nbrs))]), nbr_batch_size)
        if nbr_batch_size <= 0:
            raise RuntimeError
        nbrs_batch = torch.zeros(self.hist_len, len(samples), nbr_batch_size, self.feature_size)
        nbrs_fut_batch = torch.zeros(self.fut_len, len(samples), nbr_batch_size, 2)

        # Initialize social mask batch:
        mask_batch = torch.zeros(nbr_batch_size+1, len(samples), nbr_batch_size+1, 1)

        mask_batch = mask_batch.byte()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        time_size = self.fut_len
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(time_size, len(samples), 2)
        op_mask_batch = torch.zeros(time_size, len(samples), 2)

        count = 0
        for sampleId, (hist, fut, nbrs, nbrs_fut) in enumerate(samples):
            if hist is not None:
                # Set up history, future, lateral maneuver and longitudinal maneuver batches:
                hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
                hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
                fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
                fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
                op_mask_batch[0:len(fut), sampleId, :] = 1

                # Set up neighbor, neighbor sequence length, and mask batches:
                count = 0
                for id, (nbr, nbr_fut) in enumerate(zip(nbrs, nbrs_fut)):
                    if len(nbr) != 0:
                        nbrs_batch[0:len(nbr), sampleId, count, :] = torch.from_numpy(nbr)
                        nbrs_fut_batch[0:len(nbr_fut), sampleId, count, :] = torch.from_numpy(nbr_fut)
                        mask_batch[0:len(nbr), sampleId, count, :] = 1
                        count += 1

        fut_batch = torch.cat((fut_batch.unsqueeze(2), nbrs_fut_batch), dim=2)
        mask_fut = torch.cumprod(~((fut_batch[:, :, :, 0] == 0) * (fut_batch[:, :, :, 1] == 0)), dim=0)
        hist_batch = torch.cat((hist_batch.unsqueeze(2), nbrs_batch), dim=2)
        mask_batch = (~((hist_batch[:, :, :, 0] == 0) * (hist_batch[:, :, :, 1] == 0)))
        return hist_batch, fut_batch, mask_batch, mask_fut, None, None
