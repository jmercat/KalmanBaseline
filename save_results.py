import torch
from torch.utils.data import DataLoader
import numpy as np

from losses import maskedNLL, maskedMSE

from utils import Settings, get_net, get_test_set
from kalman_prediction import KalmanCV

if __name__ == '__main__':
    args = Settings()

    net = get_net()
    # net = KalmanCV(args.dt)

    testSet = get_test_set()

    testDataloader = DataLoader(testSet, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, collate_fn=testSet.collate_fn)

    net.train_flag = False
    it_testDataloader = iter(testDataloader)
    len_test = len(it_testDataloader)
    avg_loss = 0
    hist_test = []
    fut_test = []
    pred_test = []
    proba_man_test = []
    mask_test = []
    # path_list = testSet.dataset['path']
    # for j in range(100):
    for j in range(len_test):
        hist, fut = next(it_testDataloader)
        hist = hist.to(args.device)
        fut = fut.to(args.device)

        mask = torch.cumprod(1 - (fut[:, :, 0] == 0).float() * (fut[:, :, 1] == 0).float(), dim=0)

        fut_pred = net(hist, None, fut.shape[0])[-fut.shape[0]:]

        loss = maskedNLL(fut_pred, fut, mask, 2)

        hist_test.append(hist.cpu().data.numpy())
        fut_test.append(fut.cpu().data.numpy())
        mask_test.append(mask.cpu().data.numpy())
        pred_test.append(fut_pred.cpu().data.numpy())

        avg_loss += loss.detach()
    avg_loss = avg_loss.cpu().data.numpy()

    print('Test loss:', format(avg_loss / len_test, '0.4f'))
    hist_test = np.concatenate(hist_test, axis=1).astype('float64')
    mask_test = np.concatenate(mask_test, axis=1).astype('float64')
    fut_test = np.concatenate(fut_test, axis=1).astype('float64')
    pred_test = np.concatenate(pred_test, axis=1).astype('float64')

    #np.savez_compressed('./results/' + args.load_name + '.npz', hist=hist_test,
    #                   mask=mask_test, fut=fut_test, pred=pred_test, path=path_list)
    np.savez_compressed('./results/' + args.load_name + '.npz', hist=hist_test,
                        mask=mask_test, fut=fut_test, pred=pred_test)

