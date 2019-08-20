from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py

### This is a modified version of the code : https://github.com/nachiket92/conv-social-pooling

## Helper function for log sum exp calculation:
def logsumexp(inputs, mask, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    if mask is None:
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    else:
        mask = (mask - 1)*1.e9
        outputs = s + (inputs - s + mask).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


## Custom activation for output layer (Graves, 2015)
def outputActivation(x, dim=2):
    muX = x.narrow(dim, 0, 1)
    muY = x.narrow(dim, 1, 1)
    sigX = x.narrow(dim, 2, 1)
    sigY = x.narrow(dim, 3, 1)
    rho = x.narrow(dim, 4, 1)
    sigX = torch.exp(sigX/2)
    sigY = torch.exp(sigY/2)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=dim)
    return out


### Dataset class for the NGSIM dataset
class NGSIMDataset(Dataset):

    def __init__(self, mat_traj_file, mat_tracks_file, t_h=30, t_f=50, d_s=2):
        self.D = np.array(h5py.File(mat_traj_file)['traj']).transpose()
        self.T = scp.loadmat(mat_tracks_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)

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
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        maxlen = self.t_h // self.d_s + 1

        # Initialize history, future
        time_size = self.t_f // self.d_s
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(time_size, len(samples), 2)

        for sampleId, (hist, fut) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])

        return hist_batch, fut_batch


def maskedNLL(y_pred, y_gt, mask=None, dim=3):
    eps_rho = 1e-2 #this avoids division by 0
    eps = 1e-1 #the minimum value for the standard deviation is set to 10cm to avoid overfitting of low values
    y_pred_pos = y_pred.narrow(dim, 0, 2)
    muX = y_pred_pos.narrow(dim, 0, 1)
    muY = y_pred_pos.narrow(dim, 1, 1)
    sigX = torch.clamp(y_pred.narrow(dim, 2, 1), eps, None)
    sigY = torch.clamp(y_pred.narrow(dim, 3, 1), eps, None)
    rho = torch.clamp(y_pred.narrow(dim, 4, 1), eps_rho-1, 1-eps_rho)
    ohr = 1/(1 - rho * rho)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    frac_x = diff_x / sigX
    frac_y = diff_y / sigY
    nll = 0.5 * ohr * (frac_x * frac_x + frac_y * frac_y -
                       2 * rho * frac_x * frac_y) +\
          torch.log(sigX) + torch.log(sigY) - \
          0.5 * torch.log(ohr)
    if mask is None:
        lossVal = torch.mean(nll)
    else:
        out = nll.masked_fill(mask.unsqueeze(dim) == 0, 0)
        lossVal = torch.sum(out)/torch.sum(mask)
    return lossVal


def maskedMSE(y_pred, y_gt, mask=None, dim=3):
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum((diff_x*diff_x + diff_y*diff_y)*mask.unsqueeze(dim))/torch.sum(mask)
    else:
        output = torch.mean(diff_x*diff_x + diff_y*diff_y)
    return output
