import torch
import torch.nn as nn
from predictors.kalman_basis import KalmanBasis
from predictors.constant_velocity_predictor import CV_model
from predictors.constant_acceleration_predictor import CA_model
from predictors.bicycle_predictor import Bicycle_model


class LSTMKalman(KalmanBasis):
    def __init__(self, args):
        self._feature_size = args.nn_feature_size
        self._n_layers = args.nn_n_layers
        self.command_feature = nn.Linear(self._state_size, self._feature_size)
        LSTMcells = []
        layer_norms = []
        for i in range(self._n_layers):
            LSTMcells.append(nn.LSTMCell(self._feature_size,
                                         self._feature_size))
            layer_norms.append(nn.LayerNorm(self._feature_size))
        self.LSTMcells = nn.ModuleList(LSTMcells)
        self.layer_norms = nn.ModuleList(layer_norms)

        self.command_out = nn.Linear(self._feature_size, self._n_command + ((self._state_size + 1) * self._state_size)//2)
        self.command_out.weight.data = self.command_out.weight
        nn.init.zeros_(self.command_out.bias)

    def _init_static(self, batch_size):
        super(LSTMKalman, self)._init_static(batch_size)
        device = self._H.device
        self.hx_list = [torch.zeros(batch_size, self._feature_size).to(device) for i in range(len(self.LSTMcells))]
        self.cx_list = [torch.zeros(batch_size, self._feature_size).to(device) for i in range(len(self.LSTMcells))]
        return self.hx_list, self.cx_list

    def _get_command(self, X, state=None):
        command = torch.tanh(self.command_feature(X.clone().squeeze(2)))
        if state is not None:
            self.hx_list, self.cx_list = state
        for j, (cell, l_n) in enumerate(zip(self.LSTMcells, self.layer_norms)):
            self.hx_list[j], self.cx_list[j] = cell(command, (self.hx_list[j], self.cx_list[j]))
            command = l_n(self.hx_list[j])
        command = self.command_out(command)
        command_out = torch.zeros((command.shape[0], self._n_command + self._state_size*self._state_size), device=command.device)
        command_std_vec = torch.exp(command[:, self._n_command:self._n_command+self._state_size])
        command_std_mat = torch.matmul(command_std_vec.unsqueeze(2), command_std_vec.unsqueeze(1))
        command_rho = torch.tanh(command[:, self._n_command+self._state_size:])
        counter = 0
        for i in range(self._state_size):
            for j in range(self._state_size):
                if i == j:
                    command_out[:, self._n_command + i*self._state_size + j] = command_std_mat[:, i, j]
                elif i > j:
                    command_out[:, self._n_command + i * self._state_size + j] = command_std_mat[:, i, j]*command_rho[:, counter]
                    command_out[:, self._n_command + j * self._state_size + i] = command_std_mat[:, j, i]*command_rho[:, counter]
                    counter += 1
        return command_out, (self.hx_list, self.cx_list)


# Mix-in Kalman classes with LSTM command predictions
def set_training(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

class CV_LSTM_model(LSTMKalman, CV_model):
    def __init__(self, args):
        CV_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        LSTMKalman.__init__(self, args)

class CA_LSTM_model(LSTMKalman, CA_model):
    def __init__(self, args):
        CA_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        LSTMKalman.__init__(self, args)

class Bicycle_LSTM_model(LSTMKalman, Bicycle_model):
    def __init__(self, args):
        Bicycle_model.__init__(self, args)
        if not args.train_kalman:
            set_training(self, False)
        LSTMKalman.__init__(self, args)
