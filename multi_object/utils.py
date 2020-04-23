import torch
from utils.utils import Settings
from multi_object.loadMultiObjectFusion import MultiObjectFusionDataset
from multi_object.loadMultiObjectNGSIM import MultiObjectNGSIMDataset
from multi_object.loadMultiObjectArgoverse import MultiObjectArgoverseDataset
from multi_object.multi_object_kalman import MultiObjectKalman
from predictors.bicycle_predictor import Bicycle_model
from NNpredictors.LSTM_kalman import CV_LSTM_model, CA_LSTM_model, Bicycle_LSTM_model
from predictors.constant_velocity_predictor import CV_model
from predictors.constant_acceleration_predictor import CA_model
import os, sys
import numpy as np

def xytheta2xy(h, dim):
    x = h.narrow(dim, 0, 1)
    y = h.narrow(dim, 1, 1)
    sx = h.narrow(dim, 3, 1)
    sy = h.narrow(dim, 4, 1)
    rxy = h.narrow(dim, 6, 1)
    p = h.narrow(dim, -1, 1)

    return torch.cat((x, y, sx, sy, rxy, p), dim=dim)

def xytheta2xy_np(h, axis):
    x = h.take([0], axis)
    y = h.narrow([1], axis)
    sx = h.narrow([3], axis)
    sy = h.narrow([4], axis)
    rxy = h.narrow([6], axis)
    p = h.narrow([-1], axis)

    return np.concatenate((x, y, sx, sy, rxy, p), axis=axis)


def sort_predictions(pred_fut):
    pred_fut[:, :, :, :, 5] = np.mean(pred_fut[:, :, :, :, 5], axis=0, keepdims=True)
    flat_pred_test = pred_fut.reshape([-1, 6, 6])
    flat_argsort_p = np.argsort(flat_pred_test[:, :, 5], axis=1)[:, ::-1]
    flat_pred_test_sorted_p = flat_pred_test.copy()
    for i in range(6):
        flat_pred_test_sorted_p[:, i, :] = flat_pred_test[np.arange(flat_pred_test.shape[0]), flat_argsort_p[:, i]]
    return flat_pred_test_sorted_p.reshape(
        [pred_fut.shape[0], pred_fut.shape[1], pred_fut.shape[2], pred_fut.shape[3], pred_fut.shape[4]])


def get_multi_object_dataset():
    args = Settings()
    if args.dataset == 'NGSIM':
        trSet = MultiObjectNGSIMDataset(args.NGSIM_data_directory + 'TrainSet_traj_v2.mat',
                                        args.NGSIM_data_directory + 'TrainSet_tracks_v2.mat', args=args)
        valSet = MultiObjectNGSIMDataset(args.NGSIM_data_directory + 'ValSet_traj_v2.mat',
                                         args.NGSIM_data_directory + 'ValSet_tracks_v2.mat', args=args)
    elif args.dataset == 'Argoverse':
        trSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + 'train/data', args=args)
        valSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + 'val/data', args=args)
    elif args.dataset == 'Fusion':
        trSet = MultiObjectFusionDataset(args.fusion_data_directory + 'train_sequenced_data.tar', args=args)
        valSet = MultiObjectFusionDataset(args.fusion_data_directory + 'val_sequenced_data.tar', args=args)

    return trSet, valSet


def get_multi_object_test_set():
    args = Settings()
    if args.dataset == 'NGSIM':
        testSet = MultiObjectNGSIMDataset(args.NGSIM_test_data_directory + 'TestSet_traj_v2.mat',
                                          args.NGSIM_test_data_directory + 'TestSet_tracks_v2.mat', args)
    elif args.dataset == 'Fusion':
        testSet = MultiObjectFusionDataset(args.fusion_data_directory + 'test_sequenced_data.tar', args)
    elif args.dataset == 'Argoverse':
        testSet = MultiObjectArgoverseDataset(args.argoverse_data_directory + '/val/dataset2/', False, False, True)
    else:
        raise RuntimeError('Multi object loader does not support other datasets than NGSIM and Fusion.')
    return testSet


def get_multi_object_net():
    args = Settings()

    if  args.model_type[-3:] == 'GRU':
        raise RuntimeError('The action prediction using GRU have not been implemented for multi object data.')
    if args.model_type == 'CV':
        net = MultiObjectKalman(args, CV_model)
    elif args.model_type == 'Bicycle':
        net = MultiObjectKalman(args, Bicycle_model)
    elif args.model_type == 'CA':
        net = MultiObjectKalman(args, CA_model)
    elif args.model_type == 'CV_LSTM':
        net = MultiObjectKalman(args, CV_LSTM_model)
    elif args.model_type == 'CA_LSTM':
        net = MultiObjectKalman(args, CA_LSTM_model)
    elif args.model_type == 'Bicycle_LSTM':
        net = MultiObjectKalman(args, Bicycle_LSTM_model)
    elif args.model_type == 'nn_attention':
        # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        currentdir = os.getcwd()
        module_dir = os.path.join(os.path.dirname(currentdir), 'Argoverse')
        sys.path.insert(0, module_dir)
        from attention_predictor import AttentionPredictor
        net = AttentionPredictor()
    # elif args.model_type == 'CV_GRU':
    #     net = CV_GRU_model(args)
    # elif args.model_type == 'CA_GRU':
    #     net = CA_GRU_model(args)
    # elif args.model_type == 'Bicycle_GRU':
    #     net = Bicycle_GRU_model(args)
    else:
        print('Model type ' + args.model_type + ' is not known.')

    net = net.to(args.device)

    if args.load_name != '':
        try:
            if args.model_type == 'nn_attention':
                net.load_state_dict(torch.load('./trained_models/multi_pred/' + args.model_type + '/' + args.load_name + '.tar', map_location=args.device))
            else:
                net.load_state_dict(torch.load('./trained_models/multi_objects/' + args.model_type + '/' + args.load_name + '.tar', map_location=args.device))
        except RuntimeError as err:
            print(err)
            print('Loading what can be loaded with option strict=False.')
            if args.model_type == 'nn_attention':
                net.load_state_dict(
                    torch.load('./trained_models/multi_pred/' + args.model_type + '/' + args.load_name + '.tar',
                               map_location=args.device), strict=False)
            else:
                net.load_state_dict(
                    torch.load('./trained_models/multi_objects/' + args.model_type + '/' + args.load_name + '.tar',
                               map_location=args.device), strict=False)
    return net

