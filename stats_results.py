import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

from utils.utils import Settings
from losses import simpleNLL_np, multiNLL_np

args = Settings()
# if args.dataset == 'NGSIM':
#     x_axis = 1
#     y_axis = 0
# else:
x_axis = 1
y_axis = 0

try:
    results = np.load('./results/' + args.load_name + '.npz')
    is_pickle = False
except FileNotFoundError as err:
    try:
        results = pickle.load(open('./results/' + args.load_name + '.pickle', 'rb'))
        is_pickle = True
    except FileNotFoundError as err:
        raise FileNotFoundError('Could not find the results file "' + './results/' + args.load_name +
          '.npz' + '", please run "save_multi_object_results.py" before calling "stats_results.py".')

n_pred = 1
hist_test = results['hist']
mask_test = results['mask']
fut_test = results['fut']

def sxsyrho2P_np(sxsyrho):
    shape = sxsyrho.shape[:-1]
    sxsyrho = sxsyrho.reshape([np.prod(shape), -1])
    P = np.zeros((np.prod(shape), 2, 2))
    P[:, 0, 0] = sxsyrho[:, 0]**2
    P[:, 1, 1] = sxsyrho[:, 1]**2
    P[:, 0, 1] = sxsyrho[:, 2]*sxsyrho[:, 0]*sxsyrho[:, 1]
    P[:, 1, 0] = sxsyrho[:, 2]*sxsyrho[:, 0]*sxsyrho[:, 1]

    return P.reshape((*shape, 2, 2))

if not is_pickle:
    #pred_test = results['pred'][:, :, :n_pred, :]
    pred_test = results['pred'][:, :, None, :]
    #pred_test[:, :, :, 5] = pred_test[:, :, :, 5] / np.sum(pred_test[:, :, :, 5], axis=2, keepdims=True)
    n_time_fut = results['fut'].shape[0]
    if n_pred > 1:
        pred_test[:, :, :, 5] = pred_test[:, :, :, 5] / np.sum(pred_test[:, :, :, 5], axis=2, keepdims=True)
        nll_test = multiNLL_np(pred_test, fut_test, mask_test)
    else:
        nll_test = simpleNLL_np(pred_test[:, :, 0, :], fut_test, mask_test)
    # nll_test = np.sum(simpleNLL_np(pred_test, fut_test)*mask_test, axis=1)/np.sum(mask_test, axis=1)
    err_test = fut_test[:, :, :2] - pred_test[:, :, 0, :2]
    # err_test = np.abs(fut_test[:, :, :2] - pred_test[:, :, :2])
    tiled_mask = mask_test[:, :, None]
    bias_error = np.sum(err_test*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
    bias_distance = np.sqrt(np.sum(bias_error*bias_error, axis=1))
    FDE_xy = np.sum(np.abs(err_test)*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)
    dist_error = np.sum(err_test*err_test, axis=2)*mask_test
    miss_rate = np.sum((dist_error > 4)*mask_test, axis=1)/np.sum(mask_test, axis=1)
    FDE = np.sum(np.sqrt(dist_error*mask_test), axis=1)/np.sum(mask_test, axis=1)
    rmse_test = np.sqrt(np.sum(dist_error*mask_test, axis=1)/np.sum(mask_test, axis=1))
    rmse_xy_test = np.sqrt(np.sum(err_test*err_test*tiled_mask, axis=1)/
                           np.sum(tiled_mask, axis=1))
    mean_err = np.sum((fut_test[:, :, :2] - pred_test[:, :, 0, :2])*tiled_mask, axis=1, keepdims=True)/np.sum(tiled_mask, axis=1, keepdims=True)
    # std_err_pred = np.sum(pred_test[:, :, 0, 2:4] * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)
    std_err_pred = np.sum(pred_test[:, :, 0, 2:4] * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)
    P = np.sum(sxsyrho2P_np(pred_test[:, :, 0, 2:5])*tiled_mask[:, :, :, None], axis=1) / (np.sum(tiled_mask, axis=1)[:, :, None])
    var_xy_test = np.sqrt(np.sum((err_test*err_test - mean_err)*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1))
    var_xy_test2 = np.zeros([n_time_fut,2, 2] )
    for t in range(n_time_fut):
        var_xy_test2[t, :, :] = np.cov((fut_test[t, :, :2] - pred_test[t, :, 0, :2]), rowvar=False,
                                        aweights=tiled_mask[t, :, 0])
else:
    nll_test = 0
    FDE = 0
    FDE_xy = 0
    bias_error = 0
    bias_distance = 0
    miss_rate = 0
    rmse_test = 0
    rmse_xy_test = 0
    num_seq = len(results['hist'])
    size_all = 0
    mean_err = 0
    std_err_pred = 0
    for i in range(num_seq):
        size_all += results['hist'][i].shape[1]
    for i in range(num_seq):
        batch_size = results['hist'][i].shape[1]
        coef = batch_size/size_all
        n_veh = results['hist'][i].shape[2]
        n_time_fut = results['fut'][i].shape[0]

        mask_test = results['mask'][i].reshape((n_time_fut, batch_size * n_veh))
        fut_test = results['fut'][i][:, :, :, :2].reshape((n_time_fut, batch_size * n_veh, -1))
        pred_test = results['pred'][i]
        n_pred = min(n_pred, pred_test.shape[3])

        pred_test = pred_test[:, :, :, :n_pred, :].reshape((n_time_fut, batch_size * n_veh, n_pred, -1))

        # pred_test = results['pred']
        if n_pred > 1:
            pred_test[:, :, :, 5] = pred_test[:, :, :, 5] / np.sum(pred_test[:, :, :, 5], axis=2, keepdims=True)
            nll_test += multiNLL_np(pred_test, fut_test, mask_test)*coef
        else:
            nll_test += simpleNLL_np(pred_test[:, :, 0, :], fut_test, mask_test)*coef
        # nll_test = np.sum(simpleNLL_np(pred_test, fut_test)*mask_test, axis=1)/np.sum(mask_test, axis=1)
        err_test = np.min(np.abs(fut_test[:, :, None, :2] - pred_test[:, :, :, :2]), axis=2)
        # err_test = np.abs(fut_test[:, :, :2] - pred_test[:, :, :2])
        tiled_mask = np.tile(mask_test[:, :, None], (1, 1, 2))
        bias_error += np.sum(err_test * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)*coef
        bias_distance += np.sqrt(np.sum(bias_error * bias_error, axis=1))*coef
        FDE_xy += np.sum(np.abs(err_test) * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)*coef
        dist_error = np.sum(err_test * err_test, axis=2) * mask_test
        miss_rate += np.sum((dist_error > 4) * mask_test, axis=1) / np.sum(mask_test, axis=1)*coef
        FDE += np.sum(np.sqrt(dist_error * mask_test), axis=1) / np.sum(mask_test, axis=1)*coef
        rmse_test += np.sum(dist_error * mask_test, axis=1) / np.sum(mask_test, axis=1)*coef
        rmse_xy_test += np.sum(err_test * err_test * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)*coef
        std_err_pred += np.sum(pred_test[:, :, 0, 2:4]*tiled_mask, axis=1)/np.sum(tiled_mask, axis=1)*coef

        mean_err += np.sum((fut_test[:, :, :2] - pred_test[:, :, 0, :2])*tiled_mask, axis=1, keepdims=True)/np.sum(tiled_mask, axis=1, keepdims=True)*coef

    rmse_test = np.sqrt(rmse_test)
    rmse_xy_test = np.sqrt(rmse_xy_test)
    var_xy_test = 0
    var_xy_test2 = np.zeros([n_time_fut,2, 2] )
    for i in range(num_seq):
        batch_size = results['hist'][i].shape[1]
        coef = batch_size / size_all
        n_veh = results['hist'][i].shape[2]
        n_time_fut = results['fut'][i].shape[0]

        mask_test = results['mask'][i].reshape((n_time_fut, batch_size * n_veh))
        fut_test = results['fut'][i][:, :, :, :2].reshape((n_time_fut, batch_size * n_veh, -1))
        pred_test = results['pred'][i]
        n_pred = min(n_pred, pred_test.shape[3])
        pred_test = pred_test[:, :, :, :n_pred, :].reshape((n_time_fut, batch_size * n_veh, n_pred, -1))
        tiled_mask = np.tile(mask_test[:, :, None], (1, 1, 2))
        err_test = np.min(np.abs(fut_test[:, :, None, :2] - pred_test[:, :, :, :2]), axis=2)
        var_xy_test += np.sum((err_test * err_test - mean_err) * tiled_mask, axis=1) / np.sum(tiled_mask, axis=1)*coef
        for t in range(n_time_fut):
            var_xy_test2[t, :, :] += np.cov((fut_test[t, :, :2] - pred_test[t, :, 0, :2]), rowvar=False, aweights=tiled_mask[t, :, 0])*coef

    var_xy_test = np.sqrt(var_xy_test)
std_err_test = []
std_err_pred_mean = []
indices = ((np.arange(args.time_pred) + 1)/args.dt - 1).astype('int') # every second
indices2 = ((np.arange(3) + 1)*args.time_pred/3/args.dt - 1).astype('int') # three equal time space
print(indices)
print('bias x', bias_error[indices, x_axis])
print('bias y', bias_error[indices, y_axis])
print('bias dist', bias_distance[indices])
print("bias \%", 100 * bias_distance[indices]/rmse_test[indices])
print('FDE xy', FDE_xy[indices])
print('FDE', FDE[indices])
print('rmse', rmse_test[indices])
print('nll', nll_test[indices])
print('miss rate', miss_rate[indices])
err_test_all = fut_test[:, :, None, :2] - pred_test[:, :, :, :2]
dist_error_all = np.sum(err_test_all*err_test_all, axis=3)*mask_test[:, :, None]
argmin = np.argmin(np.abs(dist_error_all[-1, :, :]), axis=1)
all_samples = np.arange(err_test_all.shape[1])
for i in indices2:
    std_err_test.append(np.cov(err_test[i, :, :], rowvar=False, aweights=mask_test[i, :]))
    pred_test_temp = pred_test[i, all_samples, argmin, 2:].astype('float64')
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

print(indices2)
scale_std = 1
ax = plt.subplot(111, aspect='equal')
rmse_line = plt.plot(rmse_xy_test[:, x_axis], rmse_xy_test[:, y_axis], color='blue', label='RMSE_xy(t)')
var_line = plt.plot(var_xy_test[:, x_axis], var_xy_test[:, y_axis], color='green', label='var_xy(t)')
# var_line2 = plt.plot(np.sqrt(var_xy_test2[:, x_axis, x_axis]), np.sqrt(var_xy_test2[:, y_axis, y_axis]), color='orange', label='var2_xy(t)')
sig_line = plt.plot(std_err_pred[:, x_axis], std_err_pred[:, y_axis], color='red', label='sig_xy(t)')
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_test[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_true = Ellipse(xy=(rmse_xy_test[index, x_axis], rmse_xy_test[index, y_axis]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_true.set_facecolor('none')
    ell_true.set_edgecolor('green')
    ax.add_artist(ell_true)
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_pred_mean[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_pred = Ellipse(xy=(rmse_xy_test[index, x_axis], rmse_xy_test[index, y_axis]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_pred.set_facecolor('none')
    ell_pred.set_edgecolor('red')
    ax.add_artist(ell_pred)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend([Line2D([0], [0], color='b', label='RMSE_xy(t)'),
            Line2D([0], [0], color='green', label='var_xy(t)'),
            # Line2D([0], [0], color='orange', label='batched_var_xy(t)'),
            Line2D([0], [0], color='red', label='sig_xy(t)'),
            ell_true, ell_pred],
           ['RMSE_xy(t)', "var_xy(t)", "sig_xy(t)", 'Global error covariance', 'Mean predicted error covariance'],
           handler_map={Ellipse: HandlerEllipse()})
plt.title('Evolution of the xy RMSE and covariance')
plt.xlim(-0.1, rmse_xy_test[-1, x_axis]+lambda_[x_axis]/(2*scale_std) + 0.1)
plt.ylim(-0.1, rmse_xy_test[-1, y_axis]+lambda_[y_axis]/(2*scale_std) + 0.1)
plt.savefig('./results/' + args.load_name + '_std')
plt.show()



scale_std = 1
ax = plt.subplot(111, aspect='equal')
x_pos = []
y_pos = []
x_pos_temp = 0
y_pos_temp = 0
for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_test[i, :, :])
    lambda_ = np.sqrt(lambda_)
    x_pos_temp += 0.5*lambda_[x_axis] / scale_std
    x_pos.append(x_pos_temp)
    y_pos.append(y_pos_temp)
    x_pos_temp += 0.5*lambda_[x_axis] / scale_std+0.5


height = lambda_[y_axis]/scale_std

for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_test[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_true = Ellipse(xy=(x_pos[i], y_pos[i]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_true.set_facecolor('none')
    ell_true.set_edgecolor('green')
    ax.add_artist(ell_true)
    plt.text(x_pos[i], y_pos[i], str((index+1)/5)+'s', horizontalalignment='center',
             verticalalignment='center')

for i, index in enumerate(indices2):
    lambda_, v = np.linalg.eig(std_err_pred_mean[i, :, :])
    lambda_ = np.sqrt(lambda_)
    ell_pred = Ellipse(xy=(x_pos[i], y_pos[i]),
                  width=lambda_[x_axis]/scale_std, height=lambda_[y_axis]/scale_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell_pred.set_facecolor('none')
    ell_pred.set_edgecolor('red')
    ax.add_artist(ell_pred)

plt.legend([ell_true, ell_pred],
           ['Global error covariance', 'Mean predicted error covariance'],
           handler_map={Ellipse: HandlerEllipse()}, loc='upper left')
plt.title('Evolution of the covariance')
plt.xlim(-0.1, x_pos_temp)
plt.ylim(-height-0.1, height+0.1)
plt.axis('off')
scalebar = ScaleBar(1, location='lower left',  height_fraction=0.01) # 1 pixel = 0.2 meter
plt.gca().add_artist(scalebar)
plt.savefig('./results/' + args.load_name + '_ellipses')
plt.show()




plt.figure(1)
hist, bins = np.histogram(dist_error[indices[1:], :].transpose(), bins=20)
logbins = np.logspace(np.log10(1.e-2), np.log10(1000), 12)
print('indices', indices)
plt.hist(dist_error[indices[1:], :].transpose(), bins=logbins, label=[str(int((i+1)/5)) for i in indices[1:]])
plt.xscale('log')
plt.legend()
plt.show()

plt.figure(1)
rmse_line = plt.plot(rmse_xy_test[:, x_axis], rmse_xy_test[:, y_axis], color='blue', label='RMSE_xy(t)')
var_line = plt.plot(var_xy_test[:, x_axis], var_xy_test[:, y_axis], color='green', label='var_xy(t)')
# var_line2 = plt.plot(np.sqrt(var_xy_test2[:, x_axis, x_axis]), np.sqrt(var_xy_test2[:, y_axis, y_axis]), color='orange', label='batched_var_xy(t)')
sig_line = plt.plot(std_err_pred[:, x_axis], std_err_pred[:, y_axis], color='red', label='sig_xy(t)')
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


