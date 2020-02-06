import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from utils import Settings
from losses import simpleNLL_np, multiNLL_np

args = Settings()
# if args.dataset == 'NGSIM':
#     x_axis = 1
#     y_axis = 0
# else:
x_axis = 0
y_axis = 1

try:
    results = np.load('./results/' + args.load_name + '.npz')
except FileNotFoundError as err:
    raise FileNotFoundError('Could not find the results file "' + './results/' + args.name +
          '.npz' + '", please run "save_results.py" before calling "stats_results.py".')

n_pred = 6
hist_test = results['hist']
mask_test = results['mask']
fut_test = results['fut']
pred_test = results['pred'][:, :, :n_pred, :]
# pred_test = results['pred']
pred_test[:, :, :, 5] = pred_test[:, :, :, 5] / np.sum(pred_test[:, :, :, 5], axis=2, keepdims=True)


nll_test = multiNLL_np(pred_test, fut_test, mask_test)
# nll_test = np.sum(simpleNLL_np(pred_test, fut_test)*mask_test, axis=1)/np.sum(mask_test, axis=1)
err_test = np.min(np.abs(fut_test[:, :, None, :2] - pred_test[:, :, :, :2]), axis=2)
# err_test = np.abs(fut_test[:, :, :2] - pred_test[:, :, :2])
tiled_mask = np.tile(mask_test[:, :, None], (1, 1, 2))
bias_error = np.sum(err_test*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
bias_distance = np.sqrt(np.sum(bias_error*bias_error, axis=1))
FDE_xy = np.sum(np.abs(err_test)*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
dist_error = np.sum(err_test*err_test, axis=2)*mask_test
miss_rate = np.sum((dist_error > 4)*mask_test, axis=1)/np.sum(mask_test, axis=1)
FDE = np.sum(np.sqrt(dist_error*mask_test), axis=1)/np.sum(mask_test, axis=1)
rmse_test = np.sqrt(np.sum(dist_error*mask_test, axis=1)/np.sum(mask_test, axis=1))
rmse_xy_test = np.sqrt(np.sum(err_test*err_test*tiled_mask, axis=1)/
                       np.sum(tiled_mask, axis=1))

std_err_test = []
std_err_pred_mean = []
indices = ((np.arange(3) + 1)*args.time_pred/3/args.dt - 1).astype('int')
indices2 = ((np.arange(3) + 1)*args.time_pred/3/args.dt - 1).astype('int')
print(indices2)
print('bias x', bias_error[indices, x_axis])
print('bias y', bias_error[indices, y_axis])
print('bias dist', bias_distance[indices])
print("bias \%", 100 * bias_distance[indices]/rmse_test[indices])
print('FDE xy', FDE_xy[indices])
print('FDE', FDE[indices])
print('rmse', rmse_test[indices])
print('nll', nll_test[indices])
print('miss rate', miss_rate[indices])
err_test_all = np.abs(fut_test[:, :, None, :2] - pred_test[:, :, :, :2])
dist_error_all = np.sum(err_test_all*err_test_all, axis=3)*mask_test[:, :, None]
argmin = np.argmin(dist_error_all[-1, :, :], axis=1)
gnegne = np.arange(err_test_all.shape[1])
for i in indices2:
    std_err_test.append(np.cov(err_test[i, :, :], rowvar=False, aweights=mask_test[i, :]))
    pred_test_temp = pred_test[i, gnegne, argmin, 2:].astype('float64')
    # pred_test_temp = pred_test[i, :, 2:].astype('float64')
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
time = np.arange(args.time_pred/args.dt)*args.dt
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
rmse_line = plt.plot(FDE_xy[:, x_axis], FDE_xy[:, y_axis], color='blue', label='RMSE_xy(t)')
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_test[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_true = Ellipse(xy=(FDE_xy[index, x_axis], FDE_xy[index, y_axis]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_true.set_facecolor('none')
    ell_true.set_edgecolor('green')
    ax.add_artist(ell_true)
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_pred_mean[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_pred = Ellipse(xy=(FDE_xy[index, x_axis], FDE_xy[index, y_axis]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_pred.set_facecolor('none')
    ell_pred.set_edgecolor('red')
    ax.add_artist(ell_pred)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Evolution of the xy FDE with global covariances at')
plt.legend([Line2D([0], [0], color='b', label='RMSE_xy(t)'),
            ell_true, ell_pred],
           ['FDE_xy(t)', 'Global error covariance', 'Mean predicted error covariance'],
           handler_map={Ellipse: HandlerEllipse()})
plt.xlim(-0.1, FDE_xy[-1, x_axis]+lambda_[x_axis]/(2*scale_std) + 0.1)
plt.ylim(-0.1, FDE_xy[-1, y_axis]+lambda_[y_axis]/(2*scale_std) + 0.1)
plt.savefig('./results/' + args.load_name + '_std')
plt.show()

plt.figure(1)
plt.hist(dist_error[indices[1:], :].transpose(), bins=20, label=[str(int(i/4)) for i in indices[1:]])
plt.legend()
plt.show()

argmax = np.argmax(dist_error[24, :])
plt.figure(3)
# plt.plot(pred_test[:, argmax, 0, x_axis], pred_test[:, argmax, 0, y_axis], '-o', color='red', label='Prediction')
# plt.plot(hist_test[:, argmax, x_axis], hist_test[:, argmax, y_axis], '-+', color='blue', label='History')
# plt.plot(fut_test[:, argmax, x_axis], fut_test[:, argmax, y_axis], '-+', color='green', label='Future')
plt.plot(pred_test[:, 0, x_axis], pred_test[:, 0, y_axis], '-o', color='red', label='Prediction')
plt.plot(hist_test[:, x_axis], hist_test[:, y_axis], '-+', color='blue', label='History')
plt.plot(fut_test[:, x_axis], fut_test[:, y_axis], '-+', color='green', label='Future')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Trajectory with observed history, future observations, and predicted future observations')
plt.legend()
plt.axis('equal')
plt.show()

# args = np.argwhere(dist_error[24, :] > 10000)
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
#     plt.savefig('Outliers\\outlier'+str(i).zfill(2))
#     plt.close()


