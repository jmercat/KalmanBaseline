import torch
from torch.utils.data import DataLoader
import torch.jit as jit
from ranger import Ranger
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from losses import maskedNLL, maskedMSE
from utils import Settings, get_dataset, get_net
from timeit import default_timer as timer

args = Settings()

logger = SummaryWriter('./logs/'+args.name)

# logger.add_hparams(args.get_dict(), {})

trSet, valSet = get_dataset()

net = get_net()

if args.optimizer == 'Ranger':
    optimizer = Ranger(net.parameters(), lr=args.lr, alpha=0.5, k=5)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

trDataloader = DataLoader(trSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=valSet.collate_fn)

# torch.autograd.set_detect_anomaly(True)

for epoch_num in range(args.n_epochs):

    it_trDataloader = iter(trDataloader)
    it_valDataloader = iter(valDataloader)

    len_tr = len(it_trDataloader)
    len_val = len(it_valDataloader)

    net.train_flag = True

    avg_nll_loss = 0
    avg_mse_loss = 0
    avg_loss = 0

    for i in range(len_tr):
        # start_time = timer()
        data = next(it_trDataloader)
        hist = data[0].to(args.device) * args.unit_conversion
        fut = data[1].to(args.device) * args.unit_conversion

        mask = torch.cumprod(1 - (fut[:, :, 0] == 0).float() * (fut[:, :, 1] == 0).float(), dim=0)
        optimizer.zero_grad()
        # data_time = timer()
        # print('Time getting data: ', data_time - start_time)
        fut_pred = net(hist, fut.shape[0])[-fut.shape[0]:]
        # pred_time = timer()
        # print('Time prediction: ', pred_time - data_time)

        mse_loss = maskedMSE(fut_pred, fut, mask, 2)
        nll_loss = maskedNLL(fut_pred, fut, mask, 2)
        if args.use_nll_loss:
            loss = nll_loss
        else:
            loss = mse_loss
        if loss != loss:
            print('Nan')
            continue
            # raise RuntimeError("The loss value is Nan.")
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        # step_time = timer()
        # print('Time backward: ', step_time - pred_time)

        avg_nll_loss += nll_loss.detach()
        avg_mse_loss += mse_loss.detach()
        avg_loss += loss.detach()
        # print('Overall step time: ', step_time - start_time)

        if i%args.print_every_n == args.print_every_n-1:
            torch.save(net.state_dict(), './trained_models/' + args.name + '.tar')
            avg_loss = avg_loss.item()
            avg_nll_loss = avg_nll_loss.item()

            print("Epoch no:", epoch_num + 1, "| Epoch progress(%):",
                  format(i / (len(trSet) / args.batch_size) * args.print_every_n, '0.2f'),
                  "| loss:", format(avg_loss / args.print_every_n, '0.4f'),
                  "| NLL:", format(avg_nll_loss / args.print_every_n, '0.4f'),
                  "| MSE:", format(avg_mse_loss / args.print_every_n, '0.4f'))
            info = {'loss': avg_loss/args.print_every_n, 'nll': avg_nll_loss/args.print_every_n, 'mse': avg_mse_loss / args.print_every_n}

            for tag, value in info.items():
                logger.add_scalar(tag, value, int((epoch_num*len_tr + i)/args.print_every_n))
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if len(param.data) > 1:
                        pass
                        # logger.add_histogram(name[1:], param.data, int((epoch_num*len_tr + i)/args.print_every_n))
                        # logger.add_histogram(name[1:] + '_grad', param.grad.data, int((epoch_num*len_tr + i)/args.print_every_n))
                    else:
                        try:
                            logger.add_scalar(name[1:], param.data.squeeze()[0], int((epoch_num * len_tr + i) / args.print_every_n))
                            # logger.add_scalar(name[1:] + '_grad', param.grad.data.squeeze()[0],
                            #                  int((epoch_num * len_tr + i) / args.print_every_n))
                        except:
                            logger.add_scalar(name[1:], param.data,
                                              int((epoch_num * len_tr + i) / args.print_every_n))
                            # logger.add_scalar(name[1:] + '_grad', param.grad.data,
                            #                   int((epoch_num * len_tr + i) / args.print_every_n))
            avg_nll_loss = 0
            avg_mse_loss = 0
            avg_loss = 0

    torch.save(net.state_dict(), './trained_models/' + args.name + '.tar')
    avg_loss = 0
    net.train_flag = False
    for j in range(len_val):
        data = next(it_valDataloader)
        hist = data[0].to(args.device) * args.unit_conversion
        fut = data[1].to(args.device) * args.unit_conversion
        mask = torch.cumprod(1 - (fut[:, :, 0] == 0).float() * (fut[:, :, 1] == 0).float(), dim=0)

        fut_pred = net(hist, fut.shape[0])[-fut.shape[0]:]

        loss = maskedMSE(fut_pred, fut, mask, 2)
        avg_loss += loss.detach()
    avg_loss = avg_loss.item()

    print('Validation loss:', format(avg_loss / len_val, '0.4f'))

    info = {'val_loss': avg_loss / len_val}

    for tag, value in info.items():
        logger.add_scalar(tag, value, (epoch_num+1)*len_tr)
