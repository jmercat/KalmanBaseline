import torch
import numpy as np

from utils import Settings

args = Settings()

eps = args.std_threshold
eps_rho = args.corr_threshold


def logsumexp_np(inputs, keepdim=False):
    s, _ = np.max(inputs, axis=3, keepdims=keepdim)
    outputs = s + (inputs - s).exp().sum(axis=3, keepdims=keepdim).log()
    return outputs


def simpleNLL_np(y_pred, y_gt):
    y_pred_pos = y_pred[:, :, :2]
    muX = y_pred_pos[:, :, 0]
    muY = y_pred_pos[:, :, 1]
    sigX = np.maximum(y_pred[:, :, 2], eps)
    sigY = np.maximum(y_pred[:, :, 3], eps)
    rho = np.clip(y_pred[:, :, 4], eps_rho-1, 1-eps_rho)
    ohr = 1/(1 - rho * rho)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    diff_x = x - muX
    diff_y = y - muY
    z = ((diff_x * diff_x) / (sigX * sigX) + (diff_y * diff_y) / (sigY * sigY) -
         (2 * rho * diff_x * diff_y) / (sigX * sigY))
    nll = 0.5 * ohr * z + np.log(sigX * sigY) - 0.5*np.log(ohr) + np.log(np.pi*2)
    return nll


def simpleMSE_np(y_pred, y_gt, mask=None):
    y_pred_pos = y_pred.narrow(2, 0, 2)
    muX = y_pred_pos.narrow(2, 0, 1)
    muY = y_pred_pos.narrow(2, 1, 1)
    x = y_gt.narrow(2, 0, 1)
    y = y_gt.narrow(2, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = np.sum((diff_x*diff_x + diff_y*diff_y)*mask)/np.sum(mask)
    else:
        output = np.mean(diff_x*diff_x + diff_y*diff_y)
    return output


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


def maskedNLL(y_pred, y_gt, mask=None, dim=3):
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


def maskedBCE(pred, truth):
    torch.nn.BCELoss(pred, truth)
