

vehicle_types = {'car': 0, 'vehicle': 1, 'other': 2, 'line': 3}

class ObjectData:
    """ Simple Plain Old Data class containing id and position of an object."""
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.type = 'other'

class LineData:
    """ Simple Plain Old Data class containing id and coordinates of line keypoints."""
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.type = 'line'


class VehicleData(ObjectData):
    """ Simple Plain Old Data class containing id, position, orientation, length ond width of a vehicle."""
    def __init__(self, id, x, y, o, l, w):
        super(VehicleData, self).__init__(id, x, y)
        self.o = o
        self.l = l
        self.w = w
        self.type = 'vehicle'

class CarData(VehicleData):
    """ Simple Plain Old Data class containing position of a standard dimension car."""
    def __init__(self, id, x, y, o):
        super(CarData, self).__init__(id, x, y, o, 3.8, 1.7)
        self.type = 'car'


