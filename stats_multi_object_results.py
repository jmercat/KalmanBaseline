import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import streamlit as st
import pandas as pd

from utils import Settings
from losses import simpleNLL_np, multiNLL_np

class StatMultiObject:
    def __init__(self, args):

        time_pred = int(args.time_pred / args.dt)
        self.time_pred = time_pred
        self.dt = args.dt
        self._stats_computed = False
        self.n_class = args.n_class
        self.results = pickle.load(open('./results/' + args.load_name + '.pickle', 'rb'))

        self.n_pred = 1
        self.name_class_dict = {'car': 0, 'ego': 1, 'truck': 2, 'motorcycle': 3, 'bicycle': 4, 'pedestrian': 5}
        self.class_dict = {v: k for k, v in self.name_class_dict.items()}
        self.nll_test = np.zeros([self.n_class, time_pred])
        self.FDE = np.zeros([self.n_class, time_pred])
        self.FDE_xy = np.zeros([self.n_class, time_pred, 2])
        self.dist_error = [[[] for t in range(time_pred)] for i in range(self.n_class)]
        self.bias_error = np.zeros([self.n_class, time_pred, 2])
        self.bias_distance = np.zeros([self.n_class, time_pred])
        self.miss_rate = np.zeros([self.n_class, time_pred])
        self.rmse_test = np.zeros([self.n_class, time_pred])
        self.rmse_xy_test = np.zeros([self.n_class, time_pred, 2])
        self.n_samples = np.zeros([self.n_class, time_pred])
        self.num_seq = len(self.results['hist'])
        self.size_all = 0
        self.mean_err = np.zeros([self.n_class, time_pred, 1, 1, 2])
        self.std_err_pred = np.zeros([self.n_class, time_pred])
        for i in range(self.num_seq):
            hist_test = self.results['hist'][i]
            self.size_all += hist_test.shape[1]

    @st.cache
    def _compute_stats(self):
        if not self._stats_computed:
            for i in range(self.num_seq):
                hist_test = self.results['hist'][i]
                mask_test = self.results['mask'][i]
                fut_test = self.results['fut'][i]
                pred_test = self.results['pred'][i]
                batch_size = hist_test.shape[1]
                classes = np.median(hist_test[:, :, :, 2], axis=0, keepdims=True)
                classes[:, :, 0] = 1  # Set ego to a separate class
                coef = batch_size / self.size_all
                n_veh = hist_test.shape[2]
                n_time_fut = fut_test.shape[0]
                there = mask_test
                for c in range(self.n_class):
                    if c < self.n_class - 1:
                        class_mask = (classes == c)
                    else:
                        class_mask = (classes >= c)

                    there_class = there & class_mask
                    if there_class.any():
                        self.nll_test[c] += simpleNLL_np(pred_test[:, :, :, :, :], fut_test[:, :, :, None, :2], there_class,
                                                    4) * coef

                        err_test = np.min(np.abs(fut_test[:, :, :, None, :2] - pred_test[:, :, :, :, :2]), axis=3)
                        # err_test = np.abs(fut_test[:, :, :2] - pred_test[:, :, :2])
                        tiled_mask = np.tile(there_class[:, :, :, None], (1, 1, 1, 2))

                        denom_mean = np.sum(there_class, axis=(1, 2))
                        self.n_samples[c] += denom_mean
                        not_there_at_time = denom_mean == 0
                        denom_mean += not_there_at_time
                        denom_mean_tiled = np.sum(tiled_mask, axis=(1, 2)) + not_there_at_time[:, None]
                        self.bias_error[c] += np.sum(err_test * tiled_mask, axis=(1, 2)) / denom_mean_tiled * coef
                        self.bias_distance[c] += np.sqrt(np.sum(self.bias_error[c] * self.bias_error[c], axis=1)) * coef
                        self.FDE_xy[c] += np.sum(np.abs(err_test) * tiled_mask, axis=(1, 2)) / denom_mean_tiled * coef
                        dist_error = np.sum(err_test * err_test, axis=3) * there_class
                        # self.dist_error[c].append([dist_error[t, there_class[t]] for t in range(self.time_pred)])
                        self.dist_error[c] = [self.dist_error[c][t] + list(dist_error[t, there_class[t]]) for t in range(self.time_pred)]
                        self.miss_rate[c] += np.sum((dist_error > 4), axis=(1, 2)) / denom_mean * coef
                        self.FDE[c] += np.sum(np.sqrt(dist_error), axis=(1, 2)) / denom_mean * coef
                        self.rmse_test[c] += np.sum(dist_error, axis=(1, 2)) / denom_mean * coef
                        self.rmse_xy_test[c] += np.sum(err_test * err_test * tiled_mask, axis=(1, 2)) / denom_mean_tiled * coef
                        self.std_err_pred[c] += np.sum(pred_test[:, :, :, 0, 2] * there_class, axis=(1, 2)) / denom_mean * coef

                        self.mean_err[c] += np.sum((fut_test[:, :, :, :2] - pred_test[:, :, :, 0, :2]) * tiled_mask, axis=(1, 2),
                                              keepdims=True) / denom_mean_tiled[:, None, None, :] * coef

            self.rmse_test = np.sqrt(self.rmse_test)
            self.rmse_xy_test = np.sqrt(self.rmse_xy_test)
            self._stats_computed = True

    def _translate_object_class(self, object_class, func, *args):
        if object_class is None or object_class == 'all':
            for c in range(self.n_class):
                func(c, *args)
            return
        if object_class in self.class_dict:
            func(object_class, *args)
        elif object_class in self.name_class_dict:
            func(self.name_class_dict[object_class], *args)
        else:
            print('Cannot print unknown object class ' + str(object_class))

    def _get_indices_at_spacing(self, spacing):
        return np.arange(self.time_pred - 1, 0, -int(round(spacing / self.dt)))[::-1]

    def print_stats(self, object_class='all', spacing=1):
        if spacing < self.dt:
            print('Spacing is set to ' + str(self.dt)+'s')
            spacing = self.dt
        if spacing > self.time_pred:
            print('Only printing final values (at time '+str(self.time_pred)+'s)')
            spacing = self.time_pred
        indices = self._get_indices_at_spacing(spacing)
        self._compute_stats()

        if object_class in self.class_dict:
            print('=========== Results for ' + self.class_dict[object_class] + 's ===========')
            print('At times ', (indices+1)*self.dt)
            print('bias x', self.bias_error[object_class, indices, 0])
            print('bias y', self.bias_error[object_class, indices, 1])
            print('bias dist', self.bias_distance[object_class, indices])
            print("bias \%", 100 * self.bias_distance[object_class, indices] / self.rmse_test[object_class, indices])
            print('FDE xy', self.FDE_xy[object_class, indices])
            print('FDE', self.FDE[object_class, indices])
            print('rmse', self.rmse_test[object_class, indices])
            print('nll', self.nll_test[object_class, indices])
            print('miss rate', self.miss_rate[object_class, indices])
            print('         ==========================')
        else:
            self._translate_object_class(object_class, self.print_stats, spacing)

    def plot_hist(self, object_class='all', spacing=1):
        self._compute_stats()
        if object_class in self.class_dict:
            indices = self._get_indices_at_spacing(spacing)
            # hist, bins = np.histogram(self.self.dist_error[object_class][indices, :].transpose(), bins=20)
            logbins = np.logspace(np.log10(1.e-2), np.log10(1000), 12)
            print('indices', indices)

            plt.hist(np.array(self.dist_error[object_class])[indices, :], bins=logbins, label=[str(int((i+1)/5)) for i in indices])
            plt.xscale('log')
            plt.legend()
            plt.show()
        else:
            self._translate_object_class(object_class, self.plot_hist, spacing)

stats = StatMultiObject(Settings())
# stats.print_stats('ego', 1)
# stats.print_stats('car')
# stats.print_stats('bicycle', 1)
stats.plot_hist('ego')

