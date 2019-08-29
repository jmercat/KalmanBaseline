import torch
from torch.utils.data import DataLoader

from logger import Logger
from kalman_prediction import KalmanCV, KalmanLSTM
from loadNGSIM import NGSIMDataset, maskedNLL, maskedMSE
from ranger import Ranger

name = 'Kalman_nll'
load_name = ''
use_LSTM = False
use_nll_loss = True
n_epochs = 3
batch_size = 1024
lr = 0.01
dt = 0.2
feet_to_meters = 0.3048
logger = Logger('./logs/'+name)
if use_LSTM:
    net = KalmanLSTM(dt)
else:
    net = KalmanCV(dt)

trSet = NGSIMDataset('data/TrainSet_traj_v2.mat', 'data/TrainSet_tracks_v2.mat')
valSet = NGSIMDataset('data/ValSet_traj_v2.mat', 'data/ValSet_tracks_v2.mat')

if torch.cuda.is_available():
    net = net.cuda()
    if load_name != '':
        net.load_state_dict(torch.load('./trained_models/' + load_name + '.tar'))
else:
    if load_name != '':
        net.load_state_dict(torch.load('./trained_models/' + load_name + '.tar', map_location='cpu'))

optimizer = Ranger(net.parameters(), lr=lr)

trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=valSet.collate_fn)


for epoch_num in range(n_epochs):

    it_trDataloader = iter(trDataloader)
    it_valDataloader = iter(valDataloader)

    len_tr = len(it_trDataloader)
    len_val = len(it_valDataloader)

    net.train_flag = True

    avg_nll_loss = 0
    avg_mse_loss = 0
    avg_loss = 0

    for i in range(len_tr):
        data = next(it_trDataloader)
        if torch.cuda.is_available():
            hist = data[0].cuda() * feet_to_meters
            fut = data[1].cuda() * feet_to_meters
        else:
            hist = data[0] * feet_to_meters
            fut = data[1] * feet_to_meters

        mask = torch.cumprod(1 - (fut[:, :, 0] == 0) * (fut[:, :, 1] == 0), dim=0).float()

        fut_pred = net(hist, fut.shape[0])

        mse_loss = maskedMSE(fut_pred, fut, mask, 2)
        nll_loss = maskedNLL(fut_pred, fut, mask, 2)
        if use_nll_loss:
            loss = nll_loss
        else:
            loss = mse_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        avg_nll_loss += nll_loss.detach()
        avg_mse_loss += mse_loss.detach()
        avg_loss += loss.detach()

        if i%100 == 99:
            torch.save(net.state_dict(), './trained_models/' + name + '.tar')
            avg_loss = avg_loss.item()
            avg_nll_loss = avg_nll_loss.item()

            print("Epoch no:", epoch_num + 1, "| Epoch progress(%):",
                  format(i / (len(trSet) / batch_size) * 100, '0.2f'),
                  "| loss:", format(avg_loss / 100, '0.4f'),
                  "| NLL:", format(avg_nll_loss / 100, '0.4f'),
                  "| MSE:", format(avg_mse_loss / 100, '0.4f'))
            info = {'loss': avg_loss/100, 'nll': avg_nll_loss/100, 'mse': avg_mse_loss / 100}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, int((epoch_num*len_tr + i)/100))

            avg_nll_loss = 0
            avg_mse_loss = 0
            avg_loss = 0

    torch.save(net.state_dict(), './trained_models/' + name + '.tar')
    avg_loss = 0
    net.train_flag = False
    for j in range(len_val):
        
        data = next(it_valDataloader)
        
        hist = data[0].cuda() * feet_to_meters
        fut = data[1].cuda() * feet_to_meters
        mask = torch.cumprod(1 - (fut[:, :, 0] == 0) * (fut[:, :, 1] == 0), dim=0).float()

        fut_pred = net(hist, fut.shape[0])

        loss = maskedMSE(fut_pred, fut, mask, 2)
        avg_loss += loss.detach()
    avg_loss = avg_loss.item()

    print('Validation loss:', format(avg_loss / len_val, '0.4f'))

    info = {'val_loss': avg_loss / len_val}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, (epoch_num+1)*len_tr)
