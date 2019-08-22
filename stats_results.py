import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from loadNGSIM import NGSIMDataset, maskedNLL
from kalman_prediction import KalmanCV, KalmanLSTM

name = ''
batch_size = 128
model_type = 'cv' # 'lstm'

def logsumexp_np(inputs, keepdim=False):
    s, _ = np.max(inputs, axis=3, keepdims=keepdim)
    outputs = s + (inputs - s).exp().sum(axis=3, keepdims=keepdim).log()
    return outputs

def simpleNLL_np(y_pred, y_gt):
    eps = 1e-1
    eps_rho = 1e-2
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

if model_type == 'cv':
    net = KalmanCV(0.2)
elif model_type == 'lstm':
    net = KalmanLSTM(0.2)

if torch.cuda.is_available():
    net = net.cuda()
    if name != '':
        net.load_state_dict(torch.load('./trained_models/'+name+'.tar'))
    testSet = NGSIMDataset('data/TestSet_traj_v2.mat',
                           'data/TestSet_tracks_v2.mat')
else:
    if name != '':
        net.load_state_dict(torch.load('./trained_models/'+name+'.tar', map_location='cpu'))
    testSet = NGSIMDataset('data/TestSet_traj_v2.mat',
                           'data/TestSet_tracks_v2.mat')

testDataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=testSet.collate_fn)

net.train_flag = False
it_testDataloader = iter(testDataloader)
len_test = len(it_testDataloader) # Change this to 100 for a quick test
avg_loss = 0
hist_test = []
fut_test = []
pred_test = []
proba_man_test = []
mask_test = []
for j in range(len_test):
    hist, fut = next(it_testDataloader)
    if torch.cuda.is_available():
        hist = hist.cuda()
        fut = fut.cuda()

    mask = torch.cumprod(1 - (fut[:, :, 0] == 0) * (fut[:, :, 1] == 0), dim=0).float()
    hist *= 0.3048
    fut *= 0.3048

    fut_pred = net(hist, fut.shape[0])

    hist_test.append(hist.cpu().data.numpy())
    fut_test.append(fut.cpu().data.numpy())
    mask_test.append(mask.cpu().data.numpy())
    pred_test.append(fut_pred.cpu().data.numpy())

    loss = maskedNLL(fut_pred, fut, mask, 2)
    avg_loss += loss.detach()
avg_loss = avg_loss.cpu().data.numpy()

print('Test loss:', format(avg_loss / len_test, '0.4f'))
hist_test = np.concatenate(hist_test, axis=1).astype('float64')
mask_test = np.concatenate(mask_test, axis=1).astype('float64')
fut_test = np.concatenate(fut_test, axis=1).astype('float64')
pred_test = np.concatenate(pred_test, axis=1).astype('float64')
nll_test = np.sum(simpleNLL_np(pred_test, fut_test)*mask_test, axis=1)/np.sum(mask_test, axis=1)
err_test = fut_test - pred_test[:, :, :2]
tiled_mask = np.tile(mask_test[:, :, None], (1, 1, 2))
bias_error = np.sum(err_test*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
bias_distance = np.sqrt(np.sum(bias_error*bias_error, axis=1))
ADE = np.sum(np.abs(err_test)*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
dist_error = np.sum(err_test*err_test, axis=2)*mask_test
rmse_test = np.sqrt(np.sum(dist_error*mask_test, axis=1)/np.sum(mask_test, axis=1))
rmse_xy_test = np.sqrt(np.sum(err_test*err_test*tiled_mask, axis=1)/
                       np.sum(tiled_mask, axis=1))

std_err_test = []
std_err_pred_mean = []
indices = [4, 9, 14, 19, 24]
indices2 = [4, 14, 24]
print('bias x', bias_error[indices, 0])
print('bias y', bias_error[indices, 1])
print('bias dist', bias_distance[indices])
print("bias \%", 100 * bias_distance[indices]/rmse_test[indices])
print('ADE', ADE[indices])
print('rmse', rmse_test[indices])
print('nll', nll_test[indices])
for i in indices2:
    std_err_test.append(np.cov(err_test[i, :, :], rowvar=False, aweights=mask_test[i, :]))
    pred_test_temp = pred_test[i, :, 2:].astype('float64')
    pred_test_temp[:, 2] = np.prod(pred_test_temp, axis=1)
    pred_test_temp[:, :2] = pred_test_temp[:, :2]**2
    pred_test_temp2 = np.zeros((pred_test_temp.shape[0], 2, 2))
    pred_test_temp2[:, 0, 0] = pred_test_temp[:, 0]
    pred_test_temp2[:, 1, 1] = pred_test_temp[:, 1]
    pred_test_temp2[:, 0, 1] = pred_test_temp[:, 2]
    pred_test_temp2[:, 1, 0] = pred_test_temp[:, 2]
    std_err_pred_mean.append(np.mean(pred_test_temp2, axis=0))

std_err_pred_mean = np.array(std_err_pred_mean)
std_err_test = np.array(std_err_test)
time = np.arange(fut_test.shape[0])/5
# plt.figure(0)
# plt.plot(time[indices], std_err_test[:, 0], label='std err x')
# plt.plot(time[indices], std_err_pred_mean[:, 0], label='std err pred x')
# plt.plot(time[indices], std_err_pred_mean[:, 1])
# plt.plot(time, np.mean(nll_test, 1), label='nll')
# plt.plot(time, rmse_test, label='rmse')
# plt.legend()
# plt.show()


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


scale_std = 1
ax = plt.subplot(111, aspect='equal')
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_test[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_true = Ellipse(xy=(ADE[index, 1], ADE[index, 0]),
                  width=lambda_[1]/scale_std, height=lambda_[0]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_true.set_facecolor('none')
    ell_true.set_edgecolor('green')
    ax.add_artist(ell_true)
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_pred_mean[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_pred = Ellipse(xy=(ADE[index, 1], ADE[index, 0]),
                  width=lambda_[1]/scale_std, height=lambda_[0]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_pred.set_facecolor('none')
    ell_pred.set_edgecolor('red')
    ax.add_artist(ell_pred)
rmse_line = plt.plot(ADE[:, 1], ADE[:, 0], color='blue', label='RMSE_xy(t)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
# plt.title('Evolution of the xy RMSE with global covariances at')
plt.legend([Line2D([0], [0], color='b', label='RMSE_xy(t)'),
            ell_true, ell_pred],
           ['MAE_xy(t)', 'Global error covariance', 'Mean predicted error covariance'],
           handler_map={Ellipse: HandlerEllipse()})
plt.xlim(-0.1, 8.5)
plt.ylim(-0.1, 1.)
plt.show()

plt.figure(1)
plt.hist(dist_error[indices[1:], :].transpose(), bins=20, label=[str(int(i/4)) for i in indices[1:]])
plt.legend()
plt.show()

args = np.argwhere(dist_error[24, :] > 1000)
argmax = np.argmax(dist_error[24, :])
plt.figure(3)
plt.plot(hist_test[:, argmax, 1], hist_test[:, argmax, 0], '-+', color='blue', label='History')
plt.plot(pred_test[:, argmax, 1], pred_test[:, argmax, 0], '-o', color='red', label='Prediction')
plt.plot(fut_test[:, argmax, 1], fut_test[:, argmax, 0], '-+', color='green', label='Future')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Trajectory with observed history, future observations, and predicted future observations')
plt.legend()
plt.axis('equal')
plt.show()
#
# for i in args:
#     plt.figure()
#     plt.plot(hist_test[:, i, 1], hist_test[:, i, 0], '-+', color='blue', label='History')
#     plt.plot(pred_test[:, i, 1], pred_test[:, i, 0], '-o', color='red', label='Prediction')
#     plt.plot(fut_test[:, i, 1], fut_test[:, i, 0], '-+', color='green', label='Future')
#     plt.xlabel('x position (m)')
#     plt.ylabel('y position (m)')
#     plt.title('Trajectory with observed history, future observations, and predicted future observations')
#     plt.legend()
#     plt.axis('equal')
#     plt.savefig('Outliers/outlier'+str(i).zfill(2))
#     plt.close()


