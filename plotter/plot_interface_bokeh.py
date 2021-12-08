from plotter.memory import MemoryData
import os
import numpy as np
import pandas as pd
from utils.utils import Settings, get_dataset, get_net, get_test_set
from multi_object.utils import get_multi_object_net, get_multi_object_test_set, xytheta2xy_np
from plotter.plot_utils import vehicle_types, ObjectData, VehicleData, CarData
# from plotter.plot_pyplot import ScenePlotter
from plotter.plot_bokeh import ScenePlotter
# from plotter.plot_cv2 import ScenePlotter
from bokeh.models import TextAnnotation, CheckboxGroup, TextInput, Select, Button, CustomJS, RadioButtonGroup, Slider, MultiSelect
from bokeh.io import show
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.palettes import Category10
from bokeh.io import export_svgs


class PlotInterface:
    def __init__(self):

        ### Methods
        self.args = Settings()
        self.index = None
        self.data_getter = None
        self.filter = None
        self._data = None
        self._model_type = None
        self._model_dir = os.path.join(self.args.models_path, 'unique_object')
        self.controls = {}
        self.scene_plotter = ScenePlotter(self._set_head_selection)
        self.known_dir_list = ['unique_object', 'multi_object', 'multi_pred']
        self.model_types = ['mono', 'multi_obj', 'multi_pred']
        self.model_types_labels = ["Mono-object", "Multi-objects", "Multi-pred"]
        self.model_sub_types = ['CV', 'CA', 'Bicycle', 'CV_LSTM', 'CA_LSTM', 'Bicycle_LSTM',  'nn_attention']
        self.know_dataset_list = ['NGSIM', 'Argoverse', 'Fusion']

        ### initialization of the interface

        ## Model type selector

        def update_select_net():
            if self._model_dir is not None:
                file_list = [fn for fn in os.listdir(self._model_dir) if os.path.isfile(os.path.join(self._model_dir, fn))]
                file_list.sort(key=lambda fn: os.stat(os.path.join(self._model_dir, fn)).st_mtime, reverse=True)
                file_list = [os.path.splitext(fn)[0] for fn in file_list]
                self.controls['net'].options = file_list
                # print(self._model_dir)
                # print('file_list')
                # print(file_list)
                if len(file_list) > 0:
                    self.controls['net'].value = file_list[0]
                else:
                    self.controls['net'].value = None

        def update_model_type():

            self._model_type = self.model_types[self.controls['multi_mono_object'].active]
            self._model_dir = os.path.join(self.args.models_path, self.known_dir_list[self.controls['multi_mono_object'].active])
            
            existing_types = [type for type in self.model_sub_types if os.path.isdir(os.path.join(self._model_dir, type))]
            self.controls['model_sub_type'].options = existing_types
            print('existing types')
            print(existing_types)
            if len(existing_types) > 0 and not self.controls['model_sub_type'].value in existing_types:
                self.controls['model_sub_type'].value = existing_types[0]
                return
            set_model_sub_type()
            update_select_net()

        def set_model_sub_type():
            if self.controls['model_sub_type'].value is not None:
                self._model_dir = os.path.join(self._model_dir, self.controls['model_sub_type'].value)
                self.args.model_type = self.controls['model_sub_type'].value
            else:
                self._model_dir = None

        def update_multi_mono_object(attr, old, new):
            update_model_type()
            print(self._model_dir)
            self._set_data_getter()
            print('___')
        
        dir_list = [fn for fn in os.listdir(self.args.models_path) if os.path.isdir(os.path.join(self.args.models_path, fn))]
        dir_list = [value for value in dir_list if value in self.known_dir_list]
        if not dir_list:
            raise RuntimeError(f'It appears that there is no known saved models in the model path {self.args.models_path}')
        else:
            active_num = np.argwhere(dir_list[0] == np.array(self.known_dir_list))[0, 0]
        multi_mono_object = RadioButtonGroup(labels=self.model_types_labels, active=active_num)
        self.controls['multi_mono_object'] = multi_mono_object
        multi_mono_object.on_change('active', update_multi_mono_object)

        ## Model sub type selector
        dir_to_look = os.path.join(self.args.models_path, dir_list[0])
        sub_dir_list = [fn for fn in os.listdir(dir_to_look) if os.path.isdir(os.path.join(dir_to_look, fn))]
        sub_dir_list = [value for value in sub_dir_list if value in self.model_sub_types]
        if not dir_list:
            raise RuntimeError(f'It appears that there is no known saved models subtype in the model path {dir_to_look}')
        else:
            active_num = np.argwhere(sub_dir_list[0] == np.array(self.model_sub_types))[0, 0]
        model_sub_type = Select(title='Select model type:', value=self.model_sub_types[active_num], options=self.model_sub_types)
        self.controls['model_sub_type'] = model_sub_type
        model_sub_type.on_change('value', lambda att, old, new: update_model_type())

        ## Model selector
        select = Select(title="Select parameter file:", value=None, options=[])
        self.controls['net'] = select
        select.on_change('value', lambda att, old, new: self._set_data_getter())

        ## Select dataset to use
        select = Select(title='Dataset:', value=self.know_dataset_list[0], options=self.know_dataset_list)
        self.controls['dataset'] = select
        select.on_change('value', lambda att, old, new: self._set_data_getter(change_index=True))

        ## Set what to draw
        checkbox_group = CheckboxGroup(
            labels=['Draw lanes', 'Draw history', 'Draw true future', 'Draw forecast', 'Draw forecast covariance'],
            active=[0, 1, 2, 3, 4])
        self.controls['check_box'] = checkbox_group
        checkbox_group.on_change('active',
                                 lambda att, old, new: (self._update_cov(), self._update_lanes(), self._update_path()))

        ## Set the number of pred
        n_pred = Slider(start=1, end=6, step=1, value=1, title='Number of prediction hypothesis')
        self.controls['n_pred'] = n_pred
        n_pred.on_change('value', lambda att, old, new: (self._update_cov(), self._update_path()))

        ## Sequence ID input
        text_input = TextInput(title="Sequence ID to plot:", value="Random")
        self.controls['sequence_id'] = text_input

        ## Head selection input
        multi_select_head = MultiSelect(title='Attention head multiple selection:',
                                             value=[], options=[])
        self.controls['Head_selection'] = multi_select_head
        multi_select_head.on_change('value', self.scene_plotter.set_active_heads)

        ## Refresh all sample
        button = Button(label="Refresh", button_type="success")
        self.controls['refresh'] = button
        button.on_click(
            lambda event: (self._set_index(), self._set_data()))
        # button.js_on_click(CustomJS(args=dict(p=self.image), code="""p.reset.emit()"""))

        update_multi_mono_object(None, None, None)

        ## Set the interface layout
        inputs = column(*(self.controls.values()), width=320, height=1000)
        inputs.sizing_mode = "fixed"
        lay = layout([[inputs, self.scene_plotter.get_image()]], sizing_mode="scale_both")
        curdoc().add_root(lay)

        self.scene_plotter._tap_on_veh('selected', [], [0])

    def _clear_all(self):
        self.scene_plotter.clear_ellipse()
        self.scene_plotter.clear_lanes()
        self.scene_plotter.clear_pred()
        self.scene_plotter.clear_past()
        self.scene_plotter.clear_fut()
        self.scene_plotter.remove_objects()
        self.scene_plotter.clear_arrows()
        self._set_head_selection([])

    def _set_head_selection(self, options):
        self.controls['Head_selection'].options = options

    def _update_path(self):
        traj_past = self._data['past']
        traj_fut = self._data['fut']
        traj_pred = self._data['pred']
        self.scene_plotter.clear_past()
        n_vehicle = self._data['n_veh']
        active_option = [self.controls['check_box'].labels[i] for i in
                         self.controls['check_box'].active]
        self.scene_plotter.clear_pred()
        self.scene_plotter.clear_past()
        self.scene_plotter.clear_fut()
        for i in range(n_vehicle):
            if 'Draw forecast' in active_option:
                for j in range(min(self.controls['n_pred'].value, traj_pred.shape[2])):
                    mask_pred = np.logical_or(traj_pred[:, i, j, 0] != 0, traj_pred[:, i, j, 1] != 0)
                    self.scene_plotter.add_pred(traj_pred[:, i, j, :], mask=mask_pred)

            if 'Draw true future' in active_option:
                mask_fut = np.logical_or(traj_fut[:, i, 0] != 0, traj_fut[:, i, 1] != 0)
                self.scene_plotter.add_fut(
                    traj_fut[:, i, :], mask=mask_fut)

        for i in range(n_vehicle):
            if 'Draw history' in active_option:
                mask_past = np.logical_or(traj_past[:, i, 0] != 0, traj_past[:, i, 1] != 0)
                mask_past[-1] = mask_past[-2]
                self.scene_plotter.add_past(
                    traj_past[:, i, :], mask=mask_past)

    def _update_cov(self):
        n_vehicle = self._data['n_veh']
        cov = self._data['cov']
        traj_pred = self._data['pred']
        proba = self._data['probability']
        active_option = [self.controls['check_box'].labels[i] for i in
                         self.controls['check_box'].active]
        self.scene_plotter.clear_ellipse()
        if 'Draw forecast covariance' in active_option:
            for i in range(n_vehicle):
                for j in range(min(self.controls['n_pred'].value, traj_pred.shape[2])):
                    self.scene_plotter.add_ellipse(traj_pred[:, i, j, :], cov[:, i, j, :, :], proba[:, i, j])

    def _update_lanes(self):
        lanes = self._data['lanes']
        lanes_mask = self._data['lanes_mask']
        active_option = [self.controls['check_box'].labels[i] for i in
                         self.controls['check_box'].active]
        self.scene_plotter.clear_lanes()
        if lanes is not None and 'Draw lanes' in active_option:
            for i in range(lanes.shape[1]):
                self.scene_plotter.add_lane(lanes[:, i, :], lanes_mask[:, i])

    def _update_objects(self):
        traj_past = self._data['past']
        orientation = self._data['orientation']
        attention_matrix = self._data['attention']
        self.scene_plotter.remove_objects()
        n_vehicle = self._data['n_veh']
        objects = []
        for i in range(n_vehicle):
            car = CarData(0, traj_past[-1, i, 0], traj_past[-1, i, 1], orientation[i])
            objects.append(car)
        self.scene_plotter.add_objects(objects, attention_matrix)

    def _draw_image(self):
        if self._data is None:
            self._set_data()
        self._update_cov()
        self._update_lanes()
        self._update_path()
        self._update_objects()

    def _set_data_getter(self, change_index=False):
        if self.controls['net'].value is not None:
            self.args.load_name = self.controls['net'].value
            is_new_dataset = False
            if self.args.dataset != self.controls['dataset'].value:
                self.args.dataset = self.controls['dataset'].value
                is_new_dataset = True

            if self._model_type == 'multi_obj':
                self.data_getter = MemoryData(get_multi_object_net(), get_multi_object_test_set(), self.args)
            elif self._model_type == 'multi_pred':
                self.args.model_type = 'nn_attention'
                self.data_getter = MemoryData(get_multi_object_net(), get_multi_object_test_set(), self.args)
            else:
                self.data_getter = MemoryData(get_net(), get_multi_object_test_set(), self.args)
            if change_index:
                self._set_index()
            self._set_data()

    def _set_filter(self):
        # TODO : add default Kalman filters for each dataset, use them to evaluate velocity
        prev_load_name = self.args.load_name
        if self.args.dataset == 'NGSIM':
            self.args.load_name = self.args.default_NGSIM_model
            self.filter = get_net()
        elif self.args.dataset in ['Argoverse', 'Fusion']:
            print(f'No default filter for {self.args.dataset}')
            self.filter = None
        self.args.load_name = prev_load_name

    def _set_index(self):
        if self.data_getter is None:
            self._set_data_getter()
        index = self.controls['sequence_id'].value
        if index is None:
            index = np.random.randint(0, len(self.data_getter) - 1)
        elif isinstance(index, str):
            try:
                index = int(index)
            except:
                index = None
        if not isinstance(index, int) or not (0 < index < len(self.data_getter) - 1):
            index = np.random.randint(0, len(self.data_getter) - 1)
        print(index)
        self.index = index

    def _set_data(self):
        if self.index is None:
            self._set_index()
        data = self.data_getter.get_data(self.index)
        attention = self.data_getter.get_social_attention_matrix()
        traj_past, traj_fut, traj_pred = data['past'], data['fut'], data['pred']
        if traj_pred.shape[-1] == 9:
            traj_pred = xytheta2xy_np(traj_pred, 3)
        cov = np.zeros((traj_pred.shape[0], traj_pred.shape[1], traj_pred.shape[2], 2, 2))
        cov[:, :, :, 0, 0] = traj_pred[:, :, :, 2]**2
        cov[:, :, :, 1, 1] = traj_pred[:, :, :, 3]**2
        cov[:, :, :, 0, 1] = traj_pred[:, :, :, 4]*traj_pred[:, :, :, 2]*traj_pred[:, :, :, 3]
        cov[:, :, :, 1, 0] = cov[:, :, :, 0, 1]
        if self.filter is None:
            n_points_slope = 3
            vx = np.mean(traj_past[-n_points_slope:, :, 0] - traj_past[-(n_points_slope + 1):-1, :, 0], axis=0)
            vy = np.mean(traj_past[-n_points_slope:, :, 1] - traj_past[-(n_points_slope + 1):-1, :, 1], axis=0)
            orientation = np.arctan2(vy, vx)
        else:
            past, _, _, _, _, _ = self.data_getter.get_input_data(self.index)
            past_state, past_cov = self.filter.filter(past.squeeze(1))
            orientation = np.arctan2(past_state[-1, :, 3], past_state[-1, :, 2])

        if traj_pred.shape[-1] == 6:
            proba = traj_pred[:, :, :, -1]
        else:
            proba = np.ones_like(traj_pred[:, :, :, -1])

        size = traj_pred.shape[1]
        mask0_all = np.zeros(size, dtype=np.bool)
        mask0 = np.logical_or(traj_past[-1, :, 0] != 0, traj_past[-1, :, 1] != 0)
        mask0[0] = True
        n_veh = np.sum(mask0)
        mask0_all[:len(mask0)] = mask0
        if attention is not None:
            attention = attention.reshape(attention.shape[-3:])
            attention = attention[:, mask0_all, :][:, :, mask0_all]

        self._data = {'past': traj_past[:, :, :2], 'fut': traj_fut[:, :, :2], 'pred': traj_pred[:, :, :, :2], 'cov': cov,
                      'probability': proba, 'orientation': orientation, 'lanes': data['lanes'], 'lanes_mask': data['mask_lanes'],
                      'n_veh': n_veh, 'attention': attention}
        self._draw_image()


