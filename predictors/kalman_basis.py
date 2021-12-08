import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Tuple

class KalmanBasis(nn.Module):

    def __init__(self, state_size, obs_size, args):
        super(KalmanBasis, self).__init__()
        self._dt = args.dt
        self._state_size = state_size
        self._obs_size = obs_size

        self._H = nn.Parameter(torch.cat((torch.eye(obs_size), torch.zeros((obs_size, state_size - obs_size))), dim=1), requires_grad=False)
        self._H_inv = nn.Parameter(self._H.transpose(1, 0), requires_grad=False)
        self._Id = nn.Parameter(torch.eye(self._state_size), requires_grad=False)

        self.eps = args.std_threshold
        self.eps_rho = args.corr_threshold

        self._R = None

    def get_l1(self):
        return 0

    def _get_jacobian(self, X):
        pass

    def _get_Q(self, X, Q_corr):
        pass

    def _get_R(self):
        pass

    def _pred_state(self, X):
        pass

    def _init_P(self, batch_size):
        pass

    def _init_X(self, Z):
        pass

    def _init_static(self, batch_size):
        pass

    def _get_command(self, X, mask=None):
        return None, None

    def _apply_command(self, X, command):
        return 0, None

    def get_lane_attention_matrix(self):
        return None

    def get_social_attention_matrix(self):
        return None

    def _get_corr(self, X, state=None):
        command, _ = self._get_command(X, state)
        if command is not None:
            X_corr, Q_corr = self._apply_command(X, command)
            Id = torch.eye(self._state_size, requires_grad=False, device=self._H.device)
            if state is None:
                return X_corr, Q_corr + Id#torch.matmul(Q_corr, Q_corr.transpose(2, 1))
            else:
                return X_corr, Q_corr + Id, state
        else:
            return 0, None

    # @torch.jit.export
    def predict(self, X, P, X_corr, Q_corr):
        # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
        J = self._get_jacobian(X.clone())
        P = J.clone() @ P.clone() @ J.permute(0, 2, 1).clone()
        Q = self._get_Q(X, Q_corr)
        P = P + Q
        X = self._pred_state(X + X_corr)

        return X, P

    # @torch.jit.export
    def update(self, Z, X, P):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        Y = Z - self._H @ X
        R = self._get_R()
        S = self._H @ P @ self._H_inv + R

        # Solving linear equation
        # K = torch.linalg.lstsq(S.transpose(2, 1), self._H @ P.transpose(2, 1)).solution # Gradient through lstsq is not implemented
        # K = K.transpose(2, 1)
        # Inverting S
        K = P @ self._H_inv @ S.inverse()

        X = X + K @ Y

        # Classical formula
        ## P = (self._Id - K @ self._H) @ P
        # Joseph formula for stability
        ImKH = self._Id - K @ self._H
        KRK = K @ R @ K.transpose(2, 1)
        P = ImKH @ P @ ImKH.transpose(2, 1) + KRK
        return X, P

    def step(self, Z, X, P, X_corr=0, Q_corr=None):
        X, P = self.predict(X, P, X_corr, Q_corr)
        X, P = self.update(Z, X, P)
        return X, P

    def init(self, Z):
        # type: (Tensor) -> Tuple[Tensor, Tensor]
        batch_size = Z.shape[1]
        self._init_static(batch_size)
        P = self._init_P(batch_size)
        X = self._init_X(Z)
        return X, P

    def _P_to_sxsyrho(self, P):
        sigma_x = torch.sqrt(torch.clamp(P[:, 0, 0].clone(), self.eps*self.eps, None)).unsqueeze(1)
        sigma_y = torch.sqrt(torch.clamp(P[:, 1, 1].clone(), self.eps*self.eps, None)).unsqueeze(1)
        rho = torch.clamp((P[:, 0, 1] + P[:, 1, 0]).unsqueeze(1) / (2 * sigma_x * sigma_y),
                          self.eps_rho - 1, 1 - self.eps_rho)
        # if torch.mean(rho) != torch.mean(rho):
        #     print('nan P')
        #     print(P[:, 0, 1] + P[:, 1, 0])
            # assert False
        return sigma_x, sigma_y, rho

    # @torch.jit.export
    def _iter_estimate(self, inputs, X, P, results):
        # type: (Tensor, Tensor, Tensor, list) -> Tuple[Tensor, Tensor, list]
        # Estimate current state
        for i in range(1, inputs.shape[0]):
            X_corr, Q_corr = self._get_corr(X)
            X, P = self.step(inputs[i, :, :, None], X, P)
            sigma_x, sigma_y, rho = self._P_to_sxsyrho(P)
            results.append(torch.cat((X[:, :2, 0], sigma_x, sigma_y, rho), dim=1).clone())
        return X, P, results

    # @torch.jit.export
    def _iter_predict(self, num_points, X, P, results):
        # type: (int, Tensor, Tensor, list) -> Tuple[Tensor, Tensor, list]
        # Predict future states
        for i in range(num_points):
            X_corr, Q_corr = self._get_corr(X)
            X, P = self.predict(X, P, X_corr, Q_corr)
            sigma_x, sigma_y, rho = self._P_to_sxsyrho(P)
            results.append(torch.cat((X[:, :2, 0], sigma_x, sigma_y, rho), dim=1).clone())
        return X, P, results

    def forward(self, inputs, mask=None, lanes=None, lanes_mask=None, len_pred=30, keep_state=False):

        X, P = self.init(inputs)
        X, P, results = self._iter_estimate(inputs, X, P, [])
        X, P, results = self._iter_predict(len_pred, X, P, results)
        results = torch.stack(results)
        return results

    def filter(self, input):
        output_state = []
        output_cov = []
        X, P = self.init(input)
        for i in range(1, input.shape[0]):
            X, P = self.step(input[i, :, :, None], X, P)
            output_state.append((X[:, :, 0]).detach().cpu().numpy())
            output_cov.append((X[:, :, 0]).detach().cpu().numpy())
        return np.array(output_state), np.array(output_cov)


