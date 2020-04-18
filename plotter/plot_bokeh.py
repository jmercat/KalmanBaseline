import numpy as np
from bokeh.plotting import figure
from bokeh.models.glyphs import Rect, Line, Circle, Ellipse
from bokeh.models import ColumnDataSource, Arrow, NormalHead, MultiLine, Triangle
from bokeh.palettes import Category10
from bokeh.models import BoxSelectTool, TapTool
from utils import Settings
from plotter.plot_utils import vehicle_types, ObjectData, VehicleData, CarData


class ScenePlotter:
    def __init__(self, head_selection):
        self.head_selection = head_selection
        self.color_names = ['1 Blue', '2 Orange', '3 Green', '4 Red', '5 Purple', '6 Brown', '7 Pink', '8 Gray', '9 Olive', '10 Cyan']
        self.active_heads = []
        self.args = Settings()
        self.field_height = self.args.field_height
        self.field_width = self.args.field_width
        self.pixel_per_meters = 1#self.args.pixel_per_meters
        self.source_object = ColumnDataSource(data={'x': [], 'y': []})
        self.object_glyph = Circle(x='x', y='y', radius=0.5, fill_alpha=0.8, fill_color='red')
        self.source_vehicle = ColumnDataSource(data={'x': [], 'y': [], 'angle': [], 'width': [], 'height': []})

        self.source_vehicle.selected.on_change('indices', self._tap_on_veh)
        self.vehicle_glyph = Rect(x='x', y='y', angle='angle', width='width', height='height',
                                  fill_color='red', fill_alpha=0.8)
        self.image = None
        self.get_image()
        self.image_height = int(self.pixel_per_meters*self.field_height)
        self.image_width = int(self.pixel_per_meters*self.field_width)

        self.image_center = np.array([self.image_width//2, self.image_height//2])
        self.source_ellipse = []
        self.ellipse_glyph = Ellipse(x='x', y='y', width='width', height='height', angle='angle',
                                     fill_alpha='alpha', line_color=None, fill_color='blue')
        self.n_ellipses = 0

        self.source_lane = []
        self.lane_glyph = Line(x='x', y='y', line_color='gray', line_dash='dashed', line_width=3)
        self.n_lanes = 0

        self.source_arrow = [[], []]
        self.arrow_glyph = Triangle(x='x', y='y', angle='angle', size='size', line_color=None)
        self.arrow_glyph_list = []

        self.source_path_fut = []
        self.source_path_pred = []
        self.source_path_past = []

        self.fut_path_glyph = Line(x='x', y='y', line_color='green', line_width=2)
        self.pred_path_glyph = Line(x='x', y='y', line_color='red', line_width=2)
        self.past_path_glyph = Line(x='x', y='y', line_color='gray', line_width=2)

        self.n_past = 0
        self.n_fut = 0
        self.n_pred = 0
        self.attention_matrix = None

    def _arrow_glyph(self, ind, x_start, y_start, x_end, y_end, width, color_index=0):
        color = list(Category10.values())[-1][color_index]
        n_lines = len(x_end)
        x_tri = np.stack([x_start.repeat(n_lines), x_end]).transpose(1, 0)
        y_tri = np.stack([y_start.repeat(n_lines), y_end]).transpose(1, 0)
        angle = np.arctan2(y_tri[:, 1]-y_tri[:, 0], x_tri[:, 1]-x_tri[:, 0]) - np.pi/2
        size = width*2
        width = width
        x_tri = x_end
        y_tri = y_end
        triangle_source = {'x': x_tri, 'y': y_tri, 'angle': angle, 'size': size}
        if ind < len(self.source_arrow[0]):
            for i in range(min(len(self.source_arrow[0][ind]), n_lines)):
                self.arrow_glyph_list[ind][i].line_width = width[i]
                self.arrow_glyph_list[ind][i].line_color = color
                line_source = {'x': [x_start, x_end[i]], 'y': [y_start, y_end[i]]}
                self.source_arrow[0][ind][i].data = line_source
            for i in range(len(self.source_arrow[0][ind]), n_lines):
                self.arrow_glyph_list[ind].append(Line(x='x', y='y', line_width=width[i], line_color=color, line_alpha=0.7, line_cap='round'))
                line_source = {'x': [x_start, x_end[i]], 'y': [y_start, y_end[i]]}
                self.source_arrow[0][ind].append(ColumnDataSource(data=line_source))
                self.image.add_glyph(self.source_arrow[0][ind][-1], glyph=self.arrow_glyph_list[ind][-1])
            # self.source_arrow[1][ind].data = triangle_source
        else:
            source_tri = ColumnDataSource(data=triangle_source)
            self.source_arrow[1].append(source_tri)
            self.source_arrow[0].append([])
            self.arrow_glyph_list.append([])
            for i in range(n_lines):
                self.arrow_glyph_list[-1].append(Line(x='x', y='y', line_width=width[i], line_color=color, line_alpha=0.7, line_cap='round'))
                line_source = {'x': [x_start, x_end[i]], 'y': [y_start, y_end[i]]}
                self.source_arrow[0][-1].append(ColumnDataSource(data=line_source))
                self.image.add_glyph(self.source_arrow[0][-1][-1], glyph=self.arrow_glyph_list[-1][-1])
            self.arrow_glyph.fill_color = color
            # self.image.add_glyph(self.source_arrow[1][-1], glyph=self.arrow_glyph)

    def clear_arrows(self):
        for line, triangle in zip(self.source_arrow[0], self.source_arrow[1]):
            for sub_line in line:
                sub_line.data = {'x': [], 'y': []}
            triangle.data = {'x': [], 'y': [], 'angle': [], 'size': []}

    def set_active_heads(self, att, old, new):
        self.clear_arrows()
        self.active_heads = [self.color_names.index(active) for active in new]
        self._tap_on_veh('value', [], self.source_vehicle.selected.indices)

    def _tap_on_veh(self, attr, old, new):
        # Delete drawings for previous selections
        if old != []:
            self.clear_arrows()
        # Draw for each element in new selection
        coord_veh = self.source_vehicle.data
        width = self.attention_matrix
        if width is not None:
            self.head_selection(self.color_names[:width.shape[0]])
            for i, ind in enumerate(new):
                x_end = np.delete(self.source_vehicle.data['x'], ind)
                y_end = np.delete(self.source_vehicle.data['y'], ind)
                n_veh = len(x_end)
                x_start = self.source_vehicle.data['x'][ind]
                y_start = self.source_vehicle.data['y'][ind]

                for h in self.active_heads:
                    print('Attention')
                    print(self.attention_matrix[h, ind])
                    width = np.delete(self.attention_matrix[h, ind, :]*10*np.log(n_veh+1), ind)
                    self._arrow_glyph(i*len(self.active_heads) + h, x_start, y_start, x_end, y_end, width, color_index=h)
        else:
            self.head_selection([])
            print('width is None')

    def remove_objects(self):
        self.clear_arrows()
        self.source_object.data = {'x': [], 'y': []}
        self.source_vehicle.data = {'x': [], 'y': [], 'angle': [], 'width': [], 'height': []}

    def add_objects(self, objects, attention_matrix):
        # TODO: should retain objects id and use them to update the correct one
        source_object = {'x': [], 'y': []}
        source_vehicle = {'x': [], 'y': [], 'angle': [], 'width': [], 'height': []}
        self.attention_matrix = attention_matrix
        for object in objects:
            if object.type == 'other':
                source_object['x'].append(object.x)
                source_object['y'].append(object.y)
            else:
                source_vehicle['x'].append(object.x)
                source_vehicle['y'].append(object.y)
                source_vehicle['angle'].append(object.o)
                source_vehicle['width'].append(object.l)
                source_vehicle['height'].append(object.w)
        self.source_object.data = source_object
        self.source_vehicle.data = source_vehicle

    def update_objects(self, objects, attention_matrix):
        #TODO: should retain objects id and use them to update the correct one
        self.attention_matrix = attention_matrix
        count_veh = 0
        count_other = 0
        for object in objects:
            if objects.type == 'other':
                self.source_object.data['x'][count_other] = objects.x
                self.source_object.data['y'][count_other] = object.y
                count_other += 1
            else:
                self.source_vehicle.data['x'][count_veh] = object.x
                self.source_vehicle.data['y'][count_veh] = object.y
                self.source_vehicle.data['angle'][count_veh] = object.o
                self.source_vehicle.data['width'][count_veh] = object.l
                self.source_vehicle.data['height'][count_veh] = object.w
                count_veh += 1

    def add_lane(self, line, mask=None):
        if mask is not None:
            line = line[mask, :]
        source_data = {'x': line[:, 0], 'y': line[:, 1]}
        if self.n_lanes >= len(self.source_lane):
            source = ColumnDataSource(data=source_data)
            self.source_lane.append(source)
            self.image.add_glyph(self.source_lane[-1], glyph=self.lane_glyph)
        else:
            self.source_lane[self.n_lanes].data = source_data
        self.n_lanes += 1

    def clear_lanes(self):
        for source in self.source_lane:
            source.data = {'x': [], 'y': []}
        self.n_lanes = 0

    def add_past(self, path, mask=None):
        self.add_path('past', path, mask)

    def add_fut(self, path, mask=None):
        self.add_path('fut', path, mask)

    def add_pred(self, path, mask=None):
        self.add_path('pred', path, mask)

    def clear_past(self):
        self.clear_path('past')

    def clear_fut(self):
        self.clear_path('fut')

    def clear_pred(self):
        self.clear_path('pred')

    def add_path(self, type, path, mask=None):
        # print('Add path ' + type)
        if mask is not None:
            path = path[mask, :]
        source_dict = {'x': path[:, 0], 'y': path[:, 1]}
        if type == 'past':
            self._draw_path(self.n_past, self.past_path_glyph, self.source_path_past, source_dict, 'gray')
            self.n_past += 1
        elif type == 'fut':
            self._draw_path(self.n_fut, self.fut_path_glyph, self.source_path_fut, source_dict, 'green')
            self.n_fut += 1
        elif type == 'pred':
            self._draw_path(self.n_pred, self.pred_path_glyph, self.source_path_pred, source_dict, 'red')
            self.n_pred += 1
        else:
            raise RuntimeError('Unknown path type.')

    def _draw_path(self, n_path, glyph, source_list, source_dict, color):
        alpha = 0.7
        source = ColumnDataSource(data=source_dict)
        if n_path >= len(source_list):
            source_list.append(source)
            self.image.add_glyph(source_list[-1], glyph=glyph)
        else:
            source_list[n_path].data = source_dict

    def clear_path(self, type):
        # print('Clear path ' + type)
        empty_source = {'x': [], 'y': []}
        if type == 'past':
            for source in self.source_path_past:
                source.data = empty_source
            self.n_past = 0
        elif type == 'fut':
            for source in self.source_path_fut:
                source.data = empty_source
            self.n_fut = 0
        elif type == 'pred':
            for source in self.source_path_pred:
                source.data = empty_source
            self.n_pred = 0
        else:
            raise RuntimeError('Unknown path arrow type.')


    def add_ellipse(self, positions, covariances, probability, n_ellipses=5):
        scales = (2 * np.arange(n_ellipses) + 1) / n_ellipses
        n_pos = positions.shape[0]
        scales = np.tile(scales, n_pos)[:, None]

        positions = positions.repeat(n_ellipses, axis=0)
        lambda_, v = np.linalg.eig(covariances)
        angle = np.arccos(v[:, 0, 0])
        angle = angle.repeat(n_ellipses, axis=0)
        lambda_ = np.sqrt(lambda_).repeat(n_ellipses, axis=0)*scales
        alpha = np.minimum(0.5 / (np.pi * lambda_[:, 0] * lambda_[:, 1]), 1)#*probability.repeat(n_ellipses, axis=0)
        source_dict = {'x': positions[:, 0], 'y': positions[:, 1],
                       'width': lambda_[:, 0], 'height': lambda_[:, 1],
                       'angle': angle, 'alpha': alpha}
        source = ColumnDataSource(data=source_dict)

        if self.n_ellipses >= len(self.source_ellipse):
            self.source_ellipse.append(source)
            self.image.add_glyph(self.source_ellipse[-1], glyph=self.ellipse_glyph, level='underlay')
        else:
            self.source_ellipse[self.n_ellipses].data = source_dict
        self.n_ellipses += 1

    def clear_ellipse(self):
        for source in self.source_ellipse:
            source.data = {'x': [], 'y': [], 'width': [], 'height': [], 'angle': [],
                           'alpha': []}
        self.n_ellipses = 0

    def get_image(self):
        if self.image is None:
            # return figure(plot_width=self.image_width, plot_height=self.image_height)
            self.image = figure(x_axis_label='x',
                                y_axis_label='y',
                                match_aspect=True,
                                sizing_mode='stretch_both',
                                tools='wheel_zoom, box_select, tap, reset, pan',
                                )
            self.image.axis.visible = False
            self.image.add_glyph(self.source_object, glyph=self.object_glyph, level='overlay')
            self.image.add_glyph(self.source_vehicle, glyph=self.vehicle_glyph, level='overlay')
            self.image.xgrid.grid_line_color = None
            self.image.ygrid.grid_line_color = None
            self.image.toolbar.active_scroll = "auto"
        return self.image




