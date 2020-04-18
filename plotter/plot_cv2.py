import numpy as np
import cv2

from utils import Settings
from plotter.plot_utils import vehicle_types, ObjectData, VehicleData, CarData


class ScenePlotter:
    def __init__(self):
        self.args = Settings()
        self.field_height = self.args.field_height
        self.field_width = self.args.field_width
        self.pixel_per_meters = self.args.pixel_per_meters
        self.image_height = int(self.pixel_per_meters*self.field_height)
        self.image_width = int(self.pixel_per_meters*self.field_width)

        self.image_center = np.array([self.image_width//2, self.image_height//2])


    def _pixel2coor(self, pix_coor):
        if isinstance(pix_coor, list):
            return [self._pixel2coor(pc) for pc in pix_coor]
        return np.array((pix_coor - self.image_center)/self.pixel_per_meters)

    def _coor2pixel(self, coor):
        if isinstance(coor, list):
            return [self._coor2pixel(c) for c in coor]
        return np.array(self.image_center + (np.array(coor) * self.pixel_per_meters)).astype('int')

    def draw_object(self, image, object: ObjectData, color=(255, 0, 0)):
        pos = self._coor2pixel((object.x, object.y))
        if object.type == 'other':
            cv2.circle(image, pos, 0.5*self.pixel_per_meters, color)
        else:
            v1 = np.array((np.cos(object.o), np.sin(object.o)))*object.l/2*self.pixel_per_meters
            v2 = np.array((-np.sin(object.o), np.cos(object.o)))*object.w/2*self.pixel_per_meters
            rectangle = np.empty([1, 4, 2], np.int32)
            rectangle[0, 0] = tuple(np.array(pos) - v1 - v2)
            rectangle[0, 1] = tuple(np.array(pos) - v1 + v2)
            rectangle[0, 2] = tuple(np.array(pos) + v1 + v2)
            rectangle[0, 3] = tuple(np.array(pos) + v1 - v2)

            cv2.drawContours(image, [rectangle], -1, color, -1)

    def draw_line(self, image, line, mask=None, color=(0, 0, 0)):
        line = self._coor2pixel(line)
        shape = line.shape
        line = line[mask].reshape((-1, *shape[1:]))
        cv2.polylines(image, [line.astype('int32').reshape([1, -1, 2])], True, color, 1)

    def draw_ellipse(self, image, positions, covariances, n_ellipses=5, color=(0, 0, 255)):

        startAngle = 0
        endAngle = 360

        scales = (2*np.arange(n_ellipses) + 1)/n_ellipses
        positions = self._coor2pixel(positions)

        for n, (pos, cov) in enumerate(zip(positions, covariances)):
            lambda_, v = np.linalg.eig(cov)
            angle = int(np.rad2deg(np.arccos(v[0, 0])))
            lambda_ = np.sqrt(lambda_)*self.pixel_per_meters
            for i, scale in enumerate(scales):
                alpha = 30 / (np.pi * lambda_[0] * lambda_[1])
                overlay = image.copy()
                cv2.ellipse(overlay, tuple(pos), (int(lambda_[0] * scale), int(lambda_[1] * scale)), angle, startAngle,
                            endAngle,
                            color, -1)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image

    def get_image(self):
        return np.ones((self.image_height, self.image_width, 3), np.uint8) * 255

    def test(self):
        image = np.ones((self.image_height, self.image_width, 3), np.uint8) * 255
        covariance = np.random.randn(2*2*600).reshape([600, 2, 2])*10
        line = np.random.randn(2*30).reshape([30, 2])*10
        covariance = covariance.transpose(0, 2, 1) @ covariance
        centers = np.random.randn(600*2).reshape([600, 2])
        self.draw_ellipse(image, centers, covariance, 3)
        car = CarData(0, 0, 0, np.pi/8)
        self.draw_object(image, car)
        self.draw_line(image, line)
        self.show('test', image)

    def show(self, window_name, image):
        cv2.imshow('test', image)
        while True:
            ch = cv2.waitKey(1)
            quit = cv2.getWindowProperty(window_name, 0) < 0
            if ch == 27 or ch == ord('q') or ch == ord('Q') or quit:
                break


