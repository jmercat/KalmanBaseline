from __future__ import print_function
import torch
from torch.utils.data import DataLoader

from kalman_prediction import KalmanLSTM, KalmanCV
from loadNGSIM import NGSIMDataset

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    eps = 1e-3
    sigmax = np.maximum(sigmax, eps)
    sigmay = np.maximum(sigmay, eps)

    Xmu = X - mux
    Ymu = Y - muy

    rho = sigmaxy / sigmax * sigmay

    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / np.maximum(2 * (1 - rho ** 2), eps)) / denom


class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        Slider.__init__(self, *args, **kwargs)
        self.valfmt = '%d / %d'
        valinit = kwargs.get('valinit', 0)
        valmax = kwargs.get('valmax', 0)
        self.set_val(0, 0)
        self.set_val(valinit, valmax)

    def set_val(self, val, max_val=0):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            xy = self.poly.xy
            xy[2] = discrete_val, 1
            xy[3] = discrete_val, 0
            self.poly.xy = xy
            progress = self.valfmt % (int(discrete_val), int(max_val))
            self.valtext.set_text(progress)
            if self.drawon:
                self.ax.figure.canvas.draw()
            self.val = val
            if not self.eventson:
                return
            for cid, func in self.observers.items():
                func(discrete_val)

    def update_val_external(self, val, max_val):
        self.set_val(val, max_val)

## Network Arguments

load_file_name = 'Kalman_nll'
dt = 0.2
feet_to_meters = 0.3048
use_LSTM = False

# Initialize network
if use_LSTM:
    net = KalmanLSTM(dt)
else:
    net = KalmanCV(dt)

if torch.cuda.is_available():
    net = net.cuda()
    if load_file_name != '':
        net.load_state_dict(torch.load('./trained_models/' + load_file_name + '.tar'))
else:
    if load_file_name != '':
        net.load_state_dict(torch.load('./trained_models/' + load_file_name + '.tar', map_location='cpu'))

if torch.cuda.is_available():
    net = net.cuda()

data_set = NGSIMDataset('data/TestSet_traj_v2.mat',
                        'data/TestSet_tracks_v2.mat')

tsDataloader = DataLoader(data_set, batch_size=len(data_set), shuffle=True,
                          num_workers=8, collate_fn=data_set.collate_fn)

skip = 10

if torch.cuda.is_available():
    lossVals = torch.zeros(25).cuda()
    counts = torch.zeros(25).cuda()
else:
    lossVals = torch.zeros(25)
    counts = torch.zeros(25)

delta = 1.0
x_min = -200
x_max = 300
y_min = -25
y_max = 25
scale = 0.4
v_w = 5
v_l = 15


class VisualizationPlot(object):
    def __init__(self, dataset, fig=None):
        self.current_frame = 1
        self.changed_button = False
        self.plotted_objects = []
        self.data_set = dataset
        self.maximum_frames = len(dataset)

        # Create figure and axes
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(32, 4)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=1.0)
            # fig, ax = plt.subplots(figsize=((x_max-x_min)*scale,range_y*2*scale))
        else:
            self.fig = fig
            self.ax = self.fig.gca()

        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        ax_color = 'lightgoldenrodyellow'
        # Define axes for the widgets
        self.ax_slider = self.fig.add_axes([0.3, 0.11, 0.40, 0.03], facecolor=ax_color)  # Slider
        base = 0.35
        self.ax_button_previous2 = self.fig.add_axes([base + 0.00, 0.01, 0.05, 0.06])  # Previous x50 button
        self.ax_button_previous = self.fig.add_axes([base + 0.06, 0.01, 0.05, 0.06])  # Previous button
        self.ax_button_next = self.fig.add_axes([base + 0.12, 0.01, 0.05, 0.06])  # Next button
        self.ax_button_next2 = self.fig.add_axes([base + 0.18, 0.01, 0.05, 0.06])  # Next x50 button
        self.ax_random = self.fig.add_axes([base + 0.24, 0.01, 0.05, 0.06])  # Random button

        # Define the widgets
        self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', valmin=1, valmax=self.maximum_frames,
                                           valinit=self.current_frame, valfmt="%d")
        self.button_previous2 = Button(self.ax_button_previous2, 'Previous x50')
        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next2 = Button(self.ax_button_next2, 'Next x50')
        self.button_random = Button(self.ax_random, 'Random')

        # Define the callbacks for the widgets' actions
        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_random.on_clicked(self.update_button_random)
        self.scroll_event_handler = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.ax.set_autoscale_on(False)

    def trigger_update(self):
        self.remove_patches()
        self.update_figure()
        self.frame_slider.update_val_external(self.current_frame, self.maximum_frames)
        self.fig.canvas.draw_idle()

    def update_figure(self):
        # Dictionaries for the style of the different objects that are visualized
        rect_style = dict(facecolor="r", fill=True, edgecolor="k", zorder=19)
        triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19)
        text_style = dict(picker=True, size=8, color='k', zorder=10, ha="center")
        text_box_style = dict(boxstyle="round,pad=0.2", fc="yellow", alpha=.6, ec="black", lw=0.2)
        track_style = dict(color="r", linewidth=1, zorder=10)

        data = data_set[self.current_frame - 1]
        batch = ((data),)
        hist, fut = data_set.collate_fn(batch)
        self.current_frame += 1

        # Initialize Variables
        if torch.cuda.is_available():
            hist = hist.cuda()
            fut = fut.cuda()

        # Forward pass
        fut_pred = net(hist*feet_to_meters, fut.shape[0])
        fut_pred[:, :, :2] = fut_pred[:, :, :2]/feet_to_meters

        hist = hist.detach()
        fut = fut.detach()
        fut_pred = fut_pred.detach()

        Z = None
        k = 0
        plotted_objects = []

        x = np.arange(x_min, x_max, delta)
        y = np.arange(y_min, y_max, delta)
        X, Y = np.meshgrid(x, y)
        fut_pred_ = fut_pred[:, 0, :].cpu().detach().numpy()
        for i in range(fut_pred_.shape[0]):  # time
            muX = fut_pred_[i, 0]
            muY = fut_pred_[i, 1]
            sigX = fut_pred_[i, 2]
            sigY = fut_pred_[i, 3]
            rho = fut_pred_[i, 4]

            Z1 = bivariate_normal(X, Y, sigY, sigX, muY, muX, rho * sigX * sigY)

            factor = (Z1.max() - Z1.min())
            if factor < 0.00001:
                continue
            Z1 /= factor

            if Z is None:
                Z = Z1.copy()
            else:
                Z = Z + Z1

            t = self.ax.plot(fut_pred_[:, 1], fut_pred_[:, 0], 'rx-', color='red')

            plotted_objects.append(t)

        if Z is not None:
            patch = matplotlib.patches.Rectangle((-v_l * 0.5, -v_w * 0.5), v_l, v_w, color="y", fill=False)
            self.ax.add_patch(patch)  # main vehicle
            plotted_objects.append(patch)
            m = self.ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.inferno,
                               extent=(x_min, x_max, y_min - 1, y_max - 1))
            plotted_objects.append(m)

        hist_x = hist[:, k, 0].cpu().numpy()
        hist_y = hist[:, k, 1].cpu().numpy()
        t = self.ax.plot(hist_y, hist_x, 'w|-', color='orange')
        plotted_objects.append(t)

        fur_x = fut[:, k, 0].cpu().numpy()
        fur_y = fut[:, k, 1].cpu().numpy()
        t = self.ax.plot(fur_y, fur_x, 'w|-', color='green')
        plotted_objects.append(t)

        self.plotted_objects = plotted_objects

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.remove_patches()
            self.update_figure()
            self.fig.canvas.draw_idle()
        self.changed_button = False

    def update_button_previous(self, _):
        if self.current_frame > 1:
            self.current_frame = self.current_frame - 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.current_frame - 50 > 0:
            self.current_frame = self.current_frame - 50
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index lower than 1.")

    def update_button_next(self, _):
        if self.current_frame < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index higher than {}.".format(self.maximum_frames))  #

    def update_button_next2(self, _):
        if self.current_frame + 50 <= self.maximum_frames:
            self.current_frame = self.current_frame + 50
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def on_scroll(self, event):
        if event.button == 'up':
            self.update_button_previous(event)
        else:
            self.update_button_next(event)

    def update_button_random(self, _):
        self.current_frame = np.random.randint(self.maximum_frames) + 1
        self.changed_button = True
        self.trigger_update()

    def get_figure(self):
        return self.fig

    def remove_patches(self, ):
        # self.fig.canvas.mpl_disconnect('pick_event')
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()

    @staticmethod
    def show():
        plt.show()
        plt.close()


visualization_plot = VisualizationPlot(data_set)
visualization_plot.show()
