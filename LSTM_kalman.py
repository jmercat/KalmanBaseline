import torch
import torch.nn as nn
from kalman_basis import KalmanBasis
from constant_velocity_predictor import CV_model
from constant_acceleration_predictor import CA_model
from bicycle_predictor import Bicycle_model

class LSTMKalman(KalmanBasis):
    def __init__(self, args):
        self._feature_size = args.lstm_feature_size
        self._n_layers = args.lstm_n_layers
        self.command_feature = nn.Linear(self._state_size, self._feature_size)
        LSTMcells = []
        layer_norms = []
        for i in range(self._n_layers):
            LSTMcells.append(nn.LSTMCell(self._feature_size,
                                         self._feature_size))
            layer_norms.append(nn.LayerNorm(self._feature_size))
        self.LSTMcells = nn.ModuleList(LSTMcells)
        self.layer_norms = nn.ModuleList(layer_norms)

        self.command_out = nn.Linear(self._feature_size, self._n_command + self._state_size * self._state_size)

    def _init_static(self, batch_size):
        super(LSTMKalman, self)._init_static(batch_size)
        device = self._H.device
        self.hx_list = [torch.zeros(batch_size, self._feature_size).to(device) for i in range(len(self.LSTMcells))]
        self.cx_list = [torch.zeros(batch_size, self._feature_size).to(device) for i in range(len(self.LSTMcells))]

    def _get_command(self, X):
        command = torch.tanh(self.command_feature(X.clone().squeeze(2)))
        for j, (cell, l_n) in enumerate(zip(self.LSTMcells, self.layer_norms)):
            self.hx_list[j], self.cx_list[j] = cell(command, (self.hx_list[j], self.cx_list[j]))
            command = l_n(self.hx_list[j])
        command = self.command_out(command)
        return command

# Mix-in Kalman classes with LSTM command predictions

class CV_LSTM_model(LSTMKalman, CV_model):
    def __init__(self, args):
        CV_model.__init__(self, args)
        LSTMKalman.__init__(self, args)

class CA_LSTM_model(LSTMKalman, CA_model):
    def __init__(self, args):
        CA_model.__init__(self, args)
        LSTMKalman.__init__(self, args)

class Bicycle_LSTM_model(LSTMKalman, Bicycle_model):
    def __init__(self, args):
        Bicycle_model.__init__(self, args)
        LSTMKalman.__init__(self, args)
