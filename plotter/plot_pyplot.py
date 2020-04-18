import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

from utils import Settings
from plotter.plot_utils import vehicle_types, ObjectData, VehicleData, CarData


class ScenePlotter:
    def __init__(self):
        self.args = Settings()
        self.field_height = self.args.field_height
        self.field_width = self.args.field_width

    def draw_object(self, image, object: ObjectData, color=(255, 0, 0)):
        if object.type == 'other':
            object = Circle(xy=(object.x, object.y), radius=0.5, fill=True, alpha=0.8, color=np.array(color)/255)
        else:
            pos = np.array((object.x, object.y))
            v1 = np.array((np.cos(object.o), np.sin(object.o))) * object.l / 2
            v2 = np.array((-np.sin(object.o), np.cos(object.o))) * object.w / 2
            pos = pos - v1 - v2
            angle = int(np.rad2deg(object.o))
            object = Rectangle(xy=pos, width=object.l, height=object.w,
                               angle=angle, fill=True, alpha=0.8, color=np.array(color)/255)
        image.add_artist(object)

    def draw_lane(self, image, line, mask=None, color=(80, 80, 80)):
        if mask is not None:
            line = line[mask, :]
        line_artist = Line2D(xdata=line[:, 0], ydata=line[:, 1], color=np.array(color)/255, linestyle='--', linewidth=2)
        image.add_artist(line_artist)

    def draw_arrow_path(self, image, path, mask=None, color=(0, 0, 0)):
        if mask is not None:
            path = path[mask, :]
        path_artist = Line2D(xdata=path[:, 0], ydata=path[:, 1], color=np.array(color)/255, linewidth=1)
        image.add_artist(path_artist)

    def draw_ellipse(self, image, positions, covariances, n_ellipses=5, color=(0, 0, 255)):
        scales = (2*np.arange(n_ellipses) + 1)/n_ellipses
        # positions = self._coor2pixel(positions)

        #TODO: this could be vectorized...
        for n, (pos, cov) in enumerate(zip(positions, covariances)):
            lambda_, v = np.linalg.eig(cov)
            angle = int(np.rad2deg(np.arccos(v[0, 0])))
            lambda_ = np.sqrt(lambda_)
            for i, scale in enumerate(scales):
                alpha = min(1, 1 / (np.pi * lambda_[0] * lambda_[1]))
                ellipse = Ellipse(xy=pos,
                                  width=lambda_[0] * scale, height=lambda_[1] * scale,
                                  angle=angle, alpha=alpha, edgecolor=None, facecolor=np.array(color)/255)
                # ellipse.set_alpha(alpha)
                # ellipse.set_facecolor(color)
                # ellipse.set_edgecolor(None)
                image.add_artist(ellipse)
        return image

    def get_image(self):
        # return figure(plot_width=self.image_width, plot_height=self.image_height)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.figaspect(1)
        plt.xlim(-self.field_width/2, self.field_width/2)
        plt.ylim(-self.field_height/2, self.field_height/2)
        plt.axis('off')
        plt.tight_layout()
        scalebar = ScaleBar(1, location='upper right', height_fraction=0.01)  # 1 pixel = 0.2 meter
        plt.gca().add_artist(scalebar)
        return fig, ax

