import torch
import torch.nn.functional as F
from predictors.kalman_basis import KalmanBasis

class MemoryData:
    """ Keeps in memory the current sequence being predicted to compute only the update when predicting trajectories."""
    def __init__(self, predictor, dataset, args):
        self.predictor = predictor
        self.dataset = dataset
        self.max_num_veh = args.max_num_veh
        self.max_dist = args.max_dist
        self.len_mem = args.len_mem
        self.len_pred = int(args.time_pred / args.dt)
        self.current_position = None
        self.current_prediction = None
        self.next_state = None
        self.input_mem = None
        self.mask_mem = None
        self.lane_mem = None
        self.mask_lane_mem = None
        self.is_lanes = True
        self.is_mask = True
        self.is_initialized = False
        self.x_axis = 0
        self.y_axis = 1
        if args.dataset == 'NGSIM' and not args.normalize_angle:
            self.x_axis = 1
            self.y_axis = 0

    def __len__(self):
        return len(self.dataset)

    def _update_memory(self):
        # Update the memory of the last positions
        self.input_mem = torch.cat((self.input_mem[1:], self.next_state), dim=0)

        self.current_position = self.current_position + self.next_state

        # Update the mask of the memory (define existing and not existing vehicles):
        xy = self.current_position - self.current_position[:, :, 0:1, :]
        dist = torch.sum(xy * xy, dim=3)
        ##set the mask to false where no vehicle is predicted and when the predicted vehicle is too far from the
        ##center vehicle
        if self.is_mask:
            dist = torch.sum(xy * xy, dim=3)
            next_mask = (dist < self.max_dist * self.max_dist) & self.mask_mem[-1:]
            self.mask_mem = torch.cat((self.mask_mem[1:], next_mask), dim=0)

        # TODO: update lanes, this should find the lanes in the map dataset around the new position of the center vehicle
        #  maybe this should be done only once every k iteration to be faster
        # self.lane_mem = ...
        # self.mask_lane_mem = ...

    def update(self, next_state):
        if not self.is_initialized:
            print('The sequence was not initialized, cannot update it.')
            return
        self.next_state = next_state
        self._update_memory()
        if self.is_lanes:
            self.current_prediction = self.net(self.input_mem, self.mask_mem, self.lane_mem,
                                               self.mask_lane_mem, self.len_pred, self.current_position,
                                               keep_state=True)
        elif self.is_mask:
            self.current_prediction = self.predictor(self.input_mem, self.mask_mem, self.len_pred)[-self.len_pred:]
        else:
            self.current_prediction = self.predictor(self.input_mem, self.len_pred)[-self.len_pred:]

        self.current_prediction.detach()
        if self.current_prediction.ndim > 3:
            self.current_prediction = self.current_prediction.squeeze(1)
        if self.current_prediction.ndim == 3:
            self.current_prediction = self.current_prediction.unsqueeze(2)

    def init(self, input_traj, mask_input=None, lane_input=None, lane_mask=None):
        if mask_input is None:
            self.is_mask = False
        if lane_input is None:
            self.is_lanes = False
        else:
            if lane_mask is None:
                lane_mask = torch.ones(lane_input.shape[:-1])

        n_veh = input_traj.shape[2]
        if n_veh > self.max_num_veh:
            padded_input = input_traj[:, :, :self.max_num_veh, :]
            if self.is_mask:
                padded_mask_input = mask_input[:, :, :self.max_num_veh]
        else:
            padded_input = F.pad(input_traj, (0, 0, 0, self.max_num_veh - n_veh), mode='constant',
                                 value=0)  # pad last dim by (0, 0) second to last dim by (0, n)
            if self.is_mask:
                padded_mask_input = F.pad(mask_input, (0, self.max_num_veh - n_veh), mode='constant',
                                          value=0)  # pad last dim by (0, n)

        # Set mem:
        self.current_position = padded_input[-1:]
        self.input_mem = padded_input[-self.len_mem:]

        if self.is_mask:
            self.mask_mem = padded_mask_input[-self.len_mem:]

        if self.is_lanes:
            self.lane_mem = lane_input
            self.mask_lane_mem = lane_mask

            self.current_prediction = self.predictor(padded_input, padded_mask_input, lane_input,
                                                     lane_mask, self.len_pred,
                                                     keep_state=False)[-self.len_pred:]
        elif self.is_mask:
            self.current_prediction = self.predictor(padded_input, padded_mask_input, len_pred=self.len_pred)[-self.len_pred:]
        else:
            self.current_prediction = self.predictor(padded_input, len_pred=self.len_pred)[-self.len_pred:]

        self.current_prediction = self.current_prediction.detach()
        if self.current_prediction.ndim > 3:
            self.current_prediction = self.current_prediction.squeeze(1)
        if self.current_prediction.ndim == 3:
            self.current_prediction = self.current_prediction.unsqueeze(2)
        self.is_initialized = True

    def get_data(self, index, time=None):
        if time is not None:
            print("Index and time access to data is not handled yet.")
        past, future, mask_past, mask_fut, lanes, mask_lanes = self.dataset.collate_fn([self.dataset[index]])
        squeezed = False
        if isinstance(self.predictor, KalmanBasis):
            past = past.squeeze(1)
            future = future.squeeze(1)
            if lanes is not None:
                lanes = lanes.squeeze(1)
                mask_lanes = mask_lanes.squeeze(1)
            squeezed = True
        self.init(past, mask_past, lanes, mask_lanes)
        if not squeezed:
            past = past.squeeze(1)
            future = future.squeeze(1)
            if mask_past is not None:
                mask_past = mask_past.squeeze(1)
                mask_fut = mask_fut.squeeze(1)
            if lanes is not None:
                lanes = lanes.squeeze(1)
                mask_lanes = mask_lanes.squeeze(1)

        if lanes is not None:
            data_dict = {'past': past.detach().cpu().numpy()[:, :, [self.x_axis, self.y_axis]],
                         'fut': future.detach().cpu().numpy()[:, :, [self.x_axis, self.y_axis]],
                         'mask_past': mask_past,
                         'mask_fut': mask_fut,
                         'pred': self.get_prediction(),
                         'lanes': lanes.detach().cpu().numpy(),
                         'mask_lanes': mask_lanes.detach().cpu().numpy()}
        else:
            data_dict = {'past': past.detach().cpu().numpy()[:, :, [self.x_axis, self.y_axis]],
                         'fut': future.detach().cpu().numpy()[:, :, [self.x_axis, self.y_axis]],
                         'mask_past': mask_past,
                         'mask_fut': mask_fut,
                         'pred': self.get_prediction(),
                         'lanes': None,
                         'mask_lanes': None}
        return data_dict

    def get_input_data(self, index):
        hist_batch, fut_batch, mask_batch, mask_fut, lanes, mask_lanes = \
            self.dataset.collate_fn([self.dataset[index]])
        if lanes is not None:
            return (hist_batch[:, :, :, [self.x_axis, self.y_axis]],
                    fut_batch[:, :, :, [self.x_axis, self.y_axis]],
                    mask_batch, mask_fut,
                    lanes[:, :, [self.x_axis, self.y_axis]],
                    mask_lanes)
        else:
            return (hist_batch[:, :, :, [self.x_axis, self.y_axis]],
                    fut_batch[:, :, :, [self.x_axis, self.y_axis]],
                    mask_batch, mask_fut,
                    None, None)

    def get_prediction(self):
        if self.is_initialized:
            if self.current_prediction.shape[-1] == 5:
                indices = [self.x_axis, self.y_axis, self.x_axis + 2, self.y_axis + 2, 4]
            else:
                indices = [self.x_axis, self.y_axis, self.x_axis + 2, self.y_axis + 2, 4, 5]
            return self.current_prediction.cpu().numpy()[:, :, :, indices]
        else:
            print('No prediction returned, the data is not initialized.')
            return

    def get_social_attention_matrix(self):
        attention_matrix =self.predictor.get_social_attention_matrix()
        if attention_matrix is not None:
            return attention_matrix.detach().cpu().numpy()
