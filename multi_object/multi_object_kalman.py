import torch
import torch.nn as nn
from predictors.kalman_basis import KalmanBasis
from multi_object.multi_head_attention import MultiHeadAttention


class MultiObjectKalman(nn.Module):
    def __init__(self, args, underlying_kalman: KalmanBasis):
        super(MultiObjectKalman, self).__init__()
        self.n_class = args.n_class
        self.use_nn = args.use_nn
        self.use_class = args.use_class
        if not self.use_class:
            self.n_class = 1

        kalman = []
        for i in range(args.n_class):
            kalman.append(underlying_kalman(args))
        self.kalman = nn.ModuleList(kalman)
        self.state_size = self.kalman[0]._state_size
        if self.use_nn:
            self.n_layers = args.nn_n_layers
            self.feature_size = args.nn_feature_size
            self.interactions = MultiHeadAttention(args.nn_feature_size, head_num=args.n_head)
            self.layer_norm = nn.LayerNorm(args.nn_feature_size)
        self.social_attention_matrix = None
        self.lane_attention_matrix = None

    def get_l1(self):
        return 0

    def get_social_attention_matrix(self):
        return self.social_attention_matrix

    def get_lane_attention_matrix(self):
        return self.lane_attention_matrix

    def get_masked_state(self, state, mask):
        hx, cx = state
        masked_hx = []
        masked_cx = []
        for hx_i, cx_i in zip(hx, cx):
            masked_hx.append(hx_i[mask].clone())
            masked_cx.append(cx_i[mask].clone())
        return masked_hx, masked_cx

    def set_masked_state(self, state, masked_state, mask_class):
        hx, cx = state
        masked_hx, masked_cx = masked_state
        for i in range(len(hx)):
            hx[i][mask_class] = masked_hx[i]
            cx[i][mask_class] = masked_cx[i]
        return hx, cx

    def interact_state(self, state, mask, batch_size, n_veh):
        hx, cx = state
        hx[0] = self.interactions(hx[0].view(batch_size, n_veh, -1), mask.view(batch_size, n_veh, -1)).view(batch_size * n_veh, -1)
        hx[0] = self.layer_norm(hx[0])
        return hx, cx

    def forward(self, inputs, mask_inputs, lanes=None, mask_lanes=None, len_pred=30, keep_state=False):
        # type: (Tensor, Tensor, int) -> Tensor
        results = []
        hist_time = inputs.shape[0]
        batch_size = inputs.shape[1]
        n_veh = inputs.shape[2]
        if self.use_class:
            classes, _ = inputs[:, :, :, 2].median(dim=0, keepdim=True)
            classes[:, :, 0] = 1  # Set ego to a separate class
            classes = classes.detach()
            inputs = inputs[:, :, :, :-1].view(hist_time, batch_size * n_veh, -1)
        else:
            inputs = inputs.view(hist_time, batch_size * n_veh, -1)
        device = inputs.device
        there_before = torch.zeros(batch_size*n_veh, dtype=torch.bool, device=device)
        X = torch.zeros(batch_size*n_veh, self.state_size, 1, device=device)
        P = torch.zeros(batch_size*n_veh, self.state_size, self.state_size, device=device)
        mask_inputs = mask_inputs.view(hist_time, batch_size*n_veh)


        if self.use_nn:
            hx = [torch.zeros(batch_size*n_veh, self.feature_size).to(device) for i in range(self.n_layers)]
            cx = [torch.zeros(batch_size*n_veh, self.feature_size).to(device) for i in range(self.n_layers)]
            state = (hx, cx)

        # Kalman state estimation
        there_before = torch.zeros_like(mask_inputs[0], dtype=torch.bool)
        for i in range(hist_time):
            there = mask_inputs[i]
            for c in range(self.n_class):
                if self.use_class:
                    if c < self.n_class - 1:
                        class_mask = (classes == c).view(batch_size * n_veh)
                    else:
                        class_mask = (classes >= c).view(batch_size * n_veh)
                    there_class = there & class_mask  # tracks present at this time
                else:
                    there_class = there

                if hist_time - i <= 3: # objects that appear in the last 3 observations are not there
                    there_class = there_class & there_before
                else: # do not instantiate new objects if observed less than 3 times
                    new = there_class & ~there_before # tracks present at this time that were not there before
                    if new.any():
                        X[new], P[new] = self.kalman[c].init(inputs[i:, new, :])
                # TODO: If an object is observed then lost and observed again, it is updated from its last observed position and not from the predicted position at the time. This should be changed?
                if there_class.any():
                    if self.use_nn:
                        X_corr, Q_corr, masked_state = self.kalman[c]._get_corr(X[there_class],
                                                                                self.get_masked_state(state, there_class))
                        state = self.set_masked_state(state, masked_state, there_class)
                        X[there_class], P[there_class] = self.kalman[c].step(inputs[i, there_class, :, None],
                                                                             X[there_class], P[there_class], X_corr, Q_corr)
                    else:
                        X[there_class], P[there_class] = self.kalman[c].step(inputs[i, there_class, :, None],
                                                                             X[there_class], P[there_class], 0, None)
            if self.use_nn:
                state = self.interact_state(state, there, batch_size, n_veh)
            sigma_x, sigma_y, rho = self.kalman[0]._P_to_sxsyrho(P)
            results.append(torch.cat((X[:, :2, 0], sigma_x, sigma_y, rho), dim=1).clone())
            there_before = there_before | there

        if self.use_nn:
            self.social_attention_matrix = self.interactions.get_attention_matrix()

        # Prediction from states
        for i in range(len_pred):
            for c in range(self.n_class):
                if self.use_class:
                    if c < self.n_class - 1:
                        class_mask = (classes == c).view(batch_size * n_veh)
                    else:
                        class_mask = (classes >= c).view(batch_size * n_veh)
                    there_class = there & class_mask
                else:
                    there_class = there

                if there_class.any():
                    if self.use_nn:
                        X_corr, Q_corr, masked_state = self.kalman[c]._get_corr(X[there_class],
                                                                                self.get_masked_state(state, there_class))
                        state = self.set_masked_state(state, masked_state, there_class)
                        X[there_class], P[there_class] = self.kalman[c].predict(X[there_class],
                                                                                P[there_class], X_corr, Q_corr)
                    else:
                        X[there_class], P[there_class] = self.kalman[c].predict(X[there_class],
                                                                                P[there_class], 0, None)
                # X[there], P[there] = self.kalman.predict(X[there], P[there], 0, None)
            if self.use_nn:
                state = self.interact_state(state, there, batch_size, n_veh)
            sigma_x, sigma_y, rho = self.kalman[0]._P_to_sxsyrho(P)
            results.append(torch.cat((X[:, :2, 0], sigma_x, sigma_y, rho), dim=1).clone())

        results = torch.stack(results).view(len_pred + hist_time, batch_size, n_veh, -1)
        return results
