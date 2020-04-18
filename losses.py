import torch
import numpy as np

from utils import Settings

args = Settings()

eps = args.std_threshold
eps_rho = args.corr_threshold


# def logsumexp_np(inputs, keepdim=False):
#     s, _ = np.max(inputs, axis=3, keepdims=keepdim)
#     outputs = s + (inputs - s).exp().sum(axis=3, keepdims=keepdim).log()
#     return outputs

def logsumexp_np(inputs, mask, keepdim=False):
    s = np.max(inputs, axis=2, keepdims=True)
    if mask is None:
        outputs = s + np.log(np.sum(np.exp(inputs - s), axis=2, keepdims=True))
    else:
        mask_veh = mask[:, :, None]
        inputs_s = np.where(mask_veh == 0, -1e9, inputs - s)
        outputs = s + np.log(np.maximum(np.sum(np.exp(inputs_s), axis=2, keepdims=True), 1e-9))

    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs

def simpleNLL_np(y_pred, y_gt, mask=None, axis=2):
    y_pred_pos = y_pred.take([0, 1], axis=axis)
    muX = y_pred_pos.take(0, axis=axis)
    muY = y_pred_pos.take(1, axis=axis)
    sigX = np.maximum(y_pred.take(2, axis=axis), eps)
    sigY = np.maximum(y_pred.take(3, axis=axis), eps)
    rho = np.clip(y_pred.take(4, axis=axis), eps_rho-1, 1-eps_rho)
    ohr = 1/(1 - rho * rho)
    x = y_gt.take(0, axis=axis)
    y = y_gt.take(1, axis=axis)
    diff_x = x - muX
    diff_y = y - muY
    z = ((diff_x * diff_x) / (sigX * sigX) + (diff_y * diff_y) / (sigY * sigY) -
         (2 * rho * diff_x * diff_y) / (sigX * sigY))
    nll = 0.5 * ohr * z + np.log(sigX * sigY) - 0.5*np.log(ohr) + np.log(np.pi*2)
    if mask is None:
        nll = np.mean(nll, 1)
    else:
        if mask.shape != nll.shape:
            mask = np.expand_dims(mask, axis=-1)
        denom = np.sum(mask, tuple(range(1, nll.ndim)))
        denom += denom == 0
        nll = np.sum(nll*mask, tuple(range(1, nll.ndim))) / denom
    return nll

def multiNLL_np(y_pred, y_gt, mask=None):
    eps = 1e-1
    eps_rho = 1e-2
    # y_gt = np.tile(y_gt[:, :, None, :], (1, 1, y_pred.shape[2], 1))
    sigX = np.maximum(y_pred[:, :, :, 2], eps)
    sigY = np.maximum(y_pred[:, :, :, 3], eps)
    rho = y_pred[:, :, :, 4]
    rho = np.clip(rho, eps_rho-1, 1-eps_rho)

    ohr = 1/(1 - rho * rho)
    frac_x = (y_gt[:, :, None, 0] - y_pred[:, :, :, 0]) / sigX
    frac_y = (y_gt[:, :, None, 1] - y_pred[:, :, :, 1]) / sigY
    # norm_term = np.log(2*np.pi*sigX * sigY) - 0.5*np.log(ohr)
    ll = -0.5 * ohr * (frac_x * frac_x + frac_y * frac_y
                     - 2 * rho * frac_x * frac_y) - np.log(2*np.pi*sigX * sigY) - 0.5*np.log(ohr) + np.log(y_pred[:, :, :, 5])
    # nll = z + norm_term
    # mask = np.logical_and(mask, np.prod(nll > 0, 2))
    # ll = - nll + np.log(p)
    # ll = ll.squeeze(-1)
    nll = -logsumexp_np(ll, mask)

    if mask is None:
        lossVal = np.mean(nll, 1)
    else:
        nll = np.where(mask == 0, 0, nll)
        lossVal = np.sum(nll, 1) / np.sum(mask, 1)
    return lossVal



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
        if torch.sum(mask) > 0:
            lossVal = torch.sum(out)/torch.sum(mask)
        else:
            lossVal = torch.sum(out)
    return lossVal


def maskedMSE(y_pred, y_gt, mask=None, dim=3):
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum((diff_x*diff_x + diff_y*diff_y)*mask.unsqueeze(dim))
        if torch.sum(mask) > 0:
            output = torch.sum(output)/torch.sum(mask)
        else:
            output = torch.sum(output)
    else:
        output = torch.mean(diff_x*diff_x + diff_y*diff_y)
    return output


def maskedBCE(pred, truth):
    torch.nn.BCELoss(pred, truth)
