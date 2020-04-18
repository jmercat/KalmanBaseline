from plotter.memory import MemoryData
import os
import streamlit as st
import numpy as np
import pandas as pd
from utils import Settings, get_dataset, get_net, get_test_set
from multi_object.utils import get_multi_object_net, get_multi_object_test_set
from plotter.plot_utils import vehicle_types, ObjectData, VehicleData, CarData
# from plotter.plot_pyplot import ScenePlotter
from plotter.plot_bokeh import ScenePlotter
# from plotter.plot_cv2 import ScenePlotter


class PlotInterface:
    def __init__(self):
        st.title('Plot sample trajectories')
        self.args = Settings()
        self._get_net()
        self._get_dataset()
        self._get_filter()
        self.index = self._select_data()
        self.scene_plotter = ScenePlotter()
        self.draw_lanes = True
        self.draw_past = True
        self.draw_fut = True
        self.draw_pred = True
        self.draw_cov = True
        self._set_what_to_draw()
        self._draw_image("Road scene", "This represents the road scene, input past observation and forecasting")

    def _set_what_to_draw(self):
        self.draw_lanes = st.sidebar.checkbox('Draw lanes', self.draw_lanes)
        self.draw_past = st.sidebar.checkbox('Draw history', self.draw_past)
        self.draw_fut = st.sidebar.checkbox('Draw true future', self.draw_fut)
        self.draw_pred = st.sidebar.checkbox('Draw forecast', self.draw_pred)
        self.draw_cov = st.sidebar.checkbox('Draw forecast covariance', self.draw_cov)


    # Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
    def _draw_image(self, header, description):

        # p, image = self.scene_plotter.get_image()
        image = self.scene_plotter.get_image()
        traj_past, traj_fut, traj_pred, cov, orientation, lanes, lanes_mask = self._get_data(self.index)
        n_vehicle = traj_past.shape[1]
        if self.draw_cov:
            for i in range(n_vehicle):
                self.scene_plotter.add_ellipse(image, traj_pred[:, i, :], cov[:, i, :, :])
        else:
            self.scene_plotter.clear_ellipse()

        if lanes is not None and self.draw_lanes:
            for i in range(lanes.shape[2]):
                self.scene_plotter.add_lane(image, lanes[:, 0, i, :], lanes_mask[:, 0, i])
        else:
            self.scene_plotter.clear_lanes()

        for i in range(n_vehicle):
            if self.draw_pred:
                mask_pred = np.logical_or(traj_pred[:, i, 0] != 0, traj_pred[:, i, 1] != 0)
                self.scene_plotter.add_arrow_pred(image, traj_pred[:, i, :], mask=mask_pred, color=(200, 10, 10))
            else:
                self.scene_plotter.clear_arrow_pred()

            if self.draw_fut:
                mask_fut = np.logical_or(traj_fut[:, i, 0] != 0, traj_fut[:, i, 1] != 0)
                self.scene_plotter.add_arrow_fut(image, traj_fut[:, i, :], mask=mask_fut, color=(10, 200, 10))
            else:
                self.scene_plotter.clear_arrow_fut()

            if self.draw_past:
                mask_past = np.logical_or(traj_past[:, i, 0] != 0, traj_past[:, i, 1] != 0)
                mask_past[-1] = mask_past[-2]
                self.scene_plotter.add_arrow_past(image, traj_past[:, i, :], mask=mask_past, color=(120, 120, 120))
            else:
                self.scene_plotter.clear_arrow_past()
        objects = []
        for i in range(n_vehicle):
            objects.append(CarData(0, traj_past[-1, i, 0], traj_past[-1, i, 1], orientation[i]))
        self.scene_plotter.add_objects(image, objects)
        # Draw the header and image.
        st.subheader(header)
        st.markdown(description)
        # st.image(image)
        st.bokeh_chart(image)
        # st.pyplot(p)

    def _print_data(self):
        traj_past, traj_fut, traj_pred, orientation, lanes, lanes_mask = self._get_data(self.index)
        st.subheader('Raw data')
        st.write(traj_past)
        st.write(traj_fut)
        st.write(traj_pred)

    def _get_dataset(self):
        dataset_list = ['NGSIM', 'Argoverse', 'Fusion']
        self.args.dataset = st.sidebar.selectbox('Dataset:', dataset_list)
        self.data_getter = MemoryData(get_net(), get_multi_object_test_set(), self.args)

    def _get_net(self):
        log_dir = './logs'
        dir_list = [dI for dI in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, dI))]
        dir_list.sort(key=lambda dI: os.stat(os.path.join(log_dir, dI)).st_mtime, reverse=True)
        self.args.load_name = st.sidebar.selectbox("Select model:", dir_list)

    def _get_filter(self):
        # TODO : add default Kalman filters for each dataset, use them to evaluate velocity
        prev_load_name = self.args.load_name
        if self.args.dataset == 'NGSIM':
            self.args.load_name = 'CV_NGSIM_143_bis'
            self.filter = get_net()
        elif self.args.dataset in ['Argoverse', 'Fusion']:
            print('No default filter for ' + self.args.dataset)
            self.filter = None

        self.args.load_name = prev_load_name

    def _select_data(self):
        index_box = st.sidebar.text_input("Sequence ID to plot:", "Random")
        return self._get_index(index_box)

    def _get_index(self, index=None):
        if index is None:
            index = np.random.randint(0, len(self.data_getter) - 1)
        elif isinstance(index, str):
            try:
                index = int(index)
            except:
                index = None
        if not isinstance(index, int) or not (0 < index < len(self.data_getter) - 1):
            index = np.random.randint(0, len(self.data_getter) - 1)
        return index

    def _get_data(self, index):
        st.write("Sequence ID: %d" % index)
        data = self.data_getter.get_data(index)
        traj_past, traj_fut, traj_pred = data['past'], data['fut'], data['pred']
        cov = np.zeros((traj_pred.shape[0], traj_pred.shape[1], 2, 2))
        cov[:, :, 0, 0] = traj_pred[:, :, 2]**2
        cov[:, :, 1, 1] = traj_pred[:, :, 3]**2
        cov[:, :, 0, 1] = traj_pred[:, :, 4]*traj_pred[:, :, 2]*traj_pred[:, :, 3]
        cov[:, :, 1, 0] = cov[:, :, 0, 1]
        if self.filter is None:
            n_points_slope = 3
            vx = np.mean(traj_past[-n_points_slope:, :, 0] - traj_past[-(n_points_slope + 1):-1, :, 0], axis=0)
            vy = np.mean(traj_past[-n_points_slope:, :, 1] - traj_past[-(n_points_slope + 1):-1, :, 1], axis=0)
            orientation = np.arctan2(vy, vx)
        else:
            past, _, _, _, _, _ = self.data_getter.get_input_data(index)
            past_state, past_cov = self.filter.filter(past.squeeze(1))
            orientation = np.arctan2(past_state[-1, :, 3], past_state[-1, :, 2])
        return traj_past[:, :, :2], traj_fut[:, :, :2], traj_pred[:, :, :2], cov, orientation, data['lanes'], data['mask_lanes']


