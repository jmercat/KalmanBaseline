import torch
import torch.nn as nn
import numpy as np

from kalman_basis import KalmanBasis


class CV_model(KalmanBasis):

    def __init__(self, args,
                 position_std_x=0.09, position_std_y=3.16,
                 velocity_std_x=0.23, velocity_std_y=7.9,
                 acceleration_std_x=0.93, acceleration_std_y=2.71):
        # type: (float, float, float, float, float, float, float) -> None
        KalmanBasis.__init__(self, 4, 2, args)

        self._position_std_x = nn.Parameter(torch.ones(1) * position_std_x, requires_grad=True)
        self._position_std_y = nn.Parameter(torch.ones(1) * position_std_y, requires_grad=True)
        self._velocity_std_x = nn.Parameter(torch.ones(1) * velocity_std_x, requires_grad=True)
        self._velocity_std_y = nn.Parameter(torch.ones(1) * velocity_std_y, requires_grad=True)
        self._acceleration_std_x = nn.Parameter(torch.ones(1) * acceleration_std_x, requires_grad=True)
        self._acceleration_std_y = nn.Parameter(torch.ones(1) * acceleration_std_y, requires_grad=True)

        # coef_G = torch.randn(self._state_size, requires_grad=True)
        coef_G = torch.ones(self._state_size, requires_grad=True)
        # coef_G[0] = -1.7168
        # coef_G[1] = -0.0294
        # coef_G[2] = 0.1378
        # coef_G[3] = -0.4126
        self._coef_G = nn.Parameter(coef_G, requires_grad=True)

        # _GR = torch.eye((2, 2)) * 1e-1
        _GR = torch.eye(2) * 1e-3
        self._GR = nn.Parameter(_GR)

        self._Id = nn.Parameter(torch.eye(self._state_size), requires_grad=False)

        # Transition matrix that defines evolution of position over a time step for a given state
        self._F = nn.Parameter(torch.eye(self._state_size), requires_grad=False)
        self._F[0, 2] = args.dt
        self._F[1, 3] = args.dt

        # Command matrix that defines how commands modify the state
        self._n_command = 2
        self._B = nn.Parameter(torch.zeros(self._state_size, self._n_command), requires_grad=False) # actions are accelerations
        self._B[0, 0] = 0#args.dt * args.dt / 2
        self._B[1, 1] = 0#args.dt * args.dt / 2
        self._B[2, 0] = args.dt
        self._B[3, 1] = args.dt

    def _get_jacobian(self, X):
        return self._F.unsqueeze(0).repeat(X.shape[0], 1, 1)

    def _get_Q(self, X, Q_corr):
        # type: (Tensor) -> Tensor
        Q = self._init_Q().unsqueeze(0).repeat((X.shape[0], 1, 1))
        ax2 = self._acceleration_std_x * self._acceleration_std_x
        ay2 = self._acceleration_std_y * self._acceleration_std_y

        # acc_mat = torch.zeros((1, self._state_size, self._state_size), device=X.device)
        # acc_mat[0, 0, 0] = ax2
        # acc_mat[0, 1, 1] = ay2
        # acc_mat[0, 2, 2] = ax2
        # acc_mat[0, 3, 3] = ay2

        if Q_corr is not None:
            Q = Q_corr
        else:
            # print(Q)
            Q[:, 0, 0] *= ax2
            Q[:, 0, 2] *= ax2
            Q[:, 2, 0] *= ax2
            Q[:, 2, 2] *= ax2
            Q[:, 1, 1] *= ay2
            Q[:, 1, 3] *= ay2
            Q[:, 3, 1] *= ay2
            Q[:, 3, 3] *= ay2
        return Q

    def _get_R(self):
        R = torch.matmul(self._GR, self._GR.transpose(1, 0))
        return R

    def _pred_state(self, X):
        return torch.matmul(self._F, X)

    def _init_Q(self):
        G = torch.zeros((self._state_size, 2), requires_grad=False, device=self._H.device)
        G[0, 0] = self._dt * self._dt / 2
        G[1, 1] = self._dt * self._dt / 2
        G[2, 0] = self._dt
        G[3, 1] = self._dt
        Q = torch.matmul((G * self._coef_G.unsqueeze(1)),
                         (G * self._coef_G.unsqueeze(1)).permute(1, 0))
        # G = torch.zeros(self._state_size, requires_grad=False, device=self._H.device)
        # G[0] = self._dt * self._dt / 2
        # G[1] = self._dt * self._dt / 2
        # G[2] = self._dt
        # G[3] = self._dt
        # Q = torch.matmul((G * self._coef_G).unsqueeze(1),
        #                  (G * self._coef_G).unsqueeze(0))
        return Q

    def _init_P(self, batch_size):
        # type: (int) -> Tensor
        # R = torch.zeros((2, 2), device=self._H.device)
        # R[0, 0] = self._position_std_x ** 2
        # R[1, 1] = self._position_std_y ** 2
        # R = torch.matmul(self._GR, self._GR.transpose(1, 0))
        P = torch.zeros((batch_size, self._state_size, self._state_size), device=self._H.device)
        # P += self._H_inv @ R @ self._H
        P[:, 0, 0] = self._position_std_x * self._position_std_x
        P[:, 1, 1] = self._position_std_y * self._position_std_y
        P[:, 2, 2] = self._velocity_std_x * self._velocity_std_x
        P[:, 3, 3] = self._velocity_std_y * self._velocity_std_y
        return P

    def _init_X(self, Z):
        V = (Z[1] - Z[0]).clone()/self._dt
        X = torch.zeros((Z.shape[1], self._state_size), device=Z.device)
        X[:, 0] = Z[0, :, 0]
        X[:, 1] = Z[0, :, 1]
        X[:, 2] = V[:, 0]
        X[:, 3] = V[:, 1]
        return X[:, :, None]

    def _apply_command(self, X, command):
        u = command[:, :2]
        X_corr = torch.matmul(self._B, u.unsqueeze(2))
        return X_corr, command[:, 2:].view(command.shape[0], self._state_size, self._state_size)