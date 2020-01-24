from __future__ import print_function, division
import torch
import yaml

from bicycle_predictor import Bicycle_model
from LSTM_CV_predictor import KalmanLSTM
from constant_velocity_predictor import CV_model
from loadNGSIM import NGSIMDataset
from loadArgoverse import ArgoverseDataset
from loadFusion import FusionDataset

class Settings:
    class __Settings:
        def __init__(self):
            self.settings_dict = yaml.load(open('./settings.yaml'))
            if self.settings_dict['device'] == '':
                self.settings_dict['device'] = 'cpu'
                if torch.cuda.is_available():
                    self.settings_dict['device'] = 'cuda'
                    print('Using device ' + torch.cuda.get_device_name())
            self.settings_dict['use_yaw'] = self.settings_dict['model_type'] == 'bicycle'
            self.settings_dict['name'] = (self.settings_dict['model_type'] + '_' +
                                          self.settings_dict['dataset'] + '_' +
                                          str(self.settings_dict['training_id']))
            if self.settings_dict['dataset'] == 'NGSIM':
                self.settings_dict['dt'] = 0.2
                self.settings_dict['unit_conversion'] = 0.3048
                self.settings_dict['time_hist'] = 3
                self.settings_dict['time_pred'] = 5
            elif self.settings_dict['dataset'] == 'Argoverse':
                self.settings_dict['dt'] = 0.1
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = 3
            elif self.settings_dict['dataset'] == 'Fusion':
                self.settings_dict['dt'] = 0.04
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = 3
            else:
                raise ValueError('The dataset "' + self.settings_dict['dataset'] + '" is unknown. Please correct the'
                                 'dataset name in "settings.yaml" or modify the Settings class in "utils.py" to handle it.')

        def __str__(self):
            return repr(self) + self.settings_dict
    instance = None
    def __init__(self):
        if not Settings.instance:
            Settings.instance = Settings.__Settings()
        else:
            pass
    def __getattr__(self, name):
        return self.instance.settings_dict[name]

    def get_dict(self):
        return self.instance.settings_dict.copy()


def get_dataset():
    args = Settings()
    if args.dataset == 'NGSIM':
        trSet = NGSIMDataset( args.NGSIM_data_directory + 'TrainSet_traj_v2.mat',
                              args.NGSIM_data_directory + 'TrainSet_tracks_v2.mat', use_yaw=args.use_yaw)
        valSet = NGSIMDataset(args.NGSIM_data_directory + 'ValSet_traj_v2.mat',
                              args.NGSIM_data_directory + 'ValSet_tracks_v2.mat', use_yaw=args.use_yaw)
    elif args.dataset == 'Argoverse':
        trSet = ArgoverseDataset(args.argoverse_data_directory + 'train/data', use_yaw=args.use_yaw)
        valSet = ArgoverseDataset(args.argoverse_data_directory + 'val/data', use_yaw=args.use_yaw)
    elif args.dataset == 'Fusion':
        trSet = FusionDataset(args.fusion_data_directory + 'sequenced_data.tar', use_yaw=args.use_yaw)
        valSet = FusionDataset(args.fusion_data_directory + 'sequenced_data.tar', use_yaw=args.use_yaw)

    return trSet, valSet


def get_test_set():
    args = Settings()
    if args.dataset == 'NGSIM':
        testSet = NGSIMDataset(args.NGSIM_test_data_directory + 'TestSet_traj_v2.mat',
                             args.NGSIM_test_data_directory + 'TestSet_tracks_v2.mat', use_yaw=args.use_yaw)
    elif args.dataset == 'Argoverse':
        testSet = ArgoverseDataset(args.argoverse_data_directory + 'val/data', use_yaw=args.use_yaw)
    elif args.dataset == 'Fusion':
        testSet = FusionDataset(args.fusion_data_directory + 'sequenced_data.tar', use_yaw=args.use_yaw)

    return testSet


def get_net():
    args = Settings()
    if args.model_type == 'LSTM':
        net = KalmanLSTM(args.dt)
    elif args.model_type == 'CV':
        net = CV_model(args.dt)
    elif args.model_type == 'bicycle':
        net = Bicycle_model(args.dt)
    else:
        print('Model type ' + args.model_type + ' is not known.')

    net = net.to(args.device)

    if args.load_name != '':
        net.load_state_dict(torch.load('./trained_models/' + args.load_name + '.tar', map_location=args.device))
    return net

## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x.narrow(2, 0, 1)
    muY = x.narrow(2, 1, 1)
    sigX = x.narrow(2, 2, 1)
    sigY = x.narrow(2, 3, 1)
    rho = x.narrow(2, 4, 1)
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out

class unnormalizer():
    def __init__(self, mean, std):
        # self.mean = torch.from_numpy(mean[1:3, 0].astype('float32'))
        self.std = std.view(1, 1, 2)

    def __call__(self, x, x0=0):
        if isinstance(x, (list,)):
            return [self.__call__(item) for item in x]
        if torch.cuda.is_available():
            x[:, :, 0:2] = x[:, :, 0:2] * self.std.cuda()  # + self.mean.cuda()
            x[:, :, 0:2] = torch.cumsum(x[:, :, 0:2], dim=0) + x0
            if x.shape[2] > 2:
                x[:, :, 2:4] = x[:, :, 2:4] * self.std.cuda()
        else:
            x[:, :, 0:2] = x[:, :, 0:2] * self.std  # + self.mean
            x[:, :, 0:2] = torch.cumsum(x[:, :, 0:2], dim=0) + x0
            if x.shape[2] > 2:
                x[:, :, 2:4] = x[:, :, 2:4] * self.std
        return x

