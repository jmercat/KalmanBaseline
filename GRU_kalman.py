import torch
import torch.nn as nn
from kalman_basis import KalmanBasis
from constant_velocity_predictor import CV_model
from constant_acceleration_predictor import CA_model
from bicycle_predictor import Bicycle_model


class GRUKalman(KalmanBasis):
    def __init__(self, args):
        self._feature_size = args.nn_feature_size
        self._n_layers = args.nn_n_layers
        self.command_feature = nn.Linear(self._state_size, self._feature_size)
        GRUcells = []
        layer_norms = []
        for i in range(self._n_layers):
            GRUcells.append(nn.GRUCell(self._feature_size,
                                         self._feature_size))
            layer_norms.append(nn.LayerNorm(self._feature_size))
        self.GRUcells = nn.ModuleList(GRUcells)
        self.layer_norms = nn.ModuleList(layer_norms)

        self.command_out = nn.Linear(self._feature_size, self._n_command + self._state_size * self._state_size)
        self.command_out.weight.data = self.command_out.weight/10
        self.command_out.bias.data = self.command_out.bias*0

    def _init_static(self, batch_size):
        super(GRUKalman, self)._init_static(batch_size)
        device = self._H.device
        self.hx_list = [torch.zeros(batch_size, self._feature_size).to(device) for i in range(len(self.GRUcells))]

    def _get_command(self, X):
        command = torch.tanh(self.command_feature(X.clone().squeeze(2)))
        for j, (cell, l_n) in enumerate(zip(self.GRUcells, self.layer_norms)):
            self.hx_list[j] = cell(command, self.hx_list[j])
            command = l_n(self.hx_list[j])
        command = self.command_out(command)
        return command


# Mix-in Kalman classes with GRU command predictions
def set_training(model, train):
    for param in model.parameters():
        param.requires_grad = train

class CV_GRU_model(GRUKalman, CV_model):
    def __init__(self, args):
        CV_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        GRUKalman.__init__(self, args)

class CA_GRU_model(GRUKalman, CA_model):
    def __init__(self, args):
        CA_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        GRUKalman.__init__(self, args)

class Bicycle_GRU_model(GRUKalman, Bicycle_model):
    def __init__(self, args):
        Bicycle_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        GRUKalman.__init__(self, args)
