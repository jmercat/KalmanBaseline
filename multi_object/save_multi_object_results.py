from torch.utils.data import DataLoader

from multi_object.utils import get_multi_object_net, get_multi_object_test_set
from utils.utils import Settings
import pickle

args = Settings()

net = get_multi_object_net()

testSet = get_multi_object_test_set()

testDataloader = DataLoader(testSet, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=testSet.collate_fn)
net.train_flag = False
it_testDataloader = iter(testDataloader)
len_test = len(it_testDataloader)

output_all = []
hist_test = []
mask_test = []
fut_test = []
pred_test = []
lines_test = []
path_list = testSet.dataset['path']
# for idx, data in dataset.items():
for j in range(len_test):
    hist, fut, mask_hist, mask_fut, lines, mask_lines  = next(it_testDataloader)
    len_pred = fut.shape[0]

    hist       = hist.to(args.device)
    fut        = fut.to('cpu')
    mask_hist  = mask_hist.to(args.device)
    mask_fut   = mask_fut.to('cpu')
    lines      = lines.to('cpu')
    mask_lines = mask_lines.to('cpu')

    pred_fut = net(hist, mask_hist, len_pred)
    pred_fut = pred_fut[-len_pred:].detach().cpu().numpy()


    lines_test.append(lines)
    hist_test.append(hist.cpu().detach().numpy())
    mask_test.append(mask_fut)
    fut_test.append(fut)
    pred_test.append(pred_fut)


with open('./results/' + args.load_name + '.pickle', 'wb') as handle:
    pickle.dump({'hist':hist_test, 'mask':mask_test, 'fut':fut_test,
                 'pred':pred_test, 'path':path_list, 'lines': lines_test},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
