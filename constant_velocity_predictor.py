import torch
import torch.nn as nn
import numpy as np

from kalman_basis import KalmanBasis


class CV_model(KalmanBasis):

    def __init__(self, args,
                 position_std_x=50, position_std_y=30,
                 velocity_std_x=100, velocity_std_y=60,
                 acceleration_std_x=1, acceleration_std_y=0.5):
        # type: (float, float, float, float, float, float, float) -> None
        KalmanBasis.__init__(self, 4, 2, args)

        self._position_std_x = nn.Parameter(torch.ones(1) * position_std_x, requires_grad=True)
        self._position_std_y = nn.Parameter(torch.ones(1) * position_std_y, requires_grad=True)
        self._velocity_std_x = nn.Parameter(torch.ones(1) * velocity_std_x, requires_grad=True)
        self._velocity_std_y = nn.Parameter(torch.ones(1) * velocity_std_y, requires_grad=True)
        self._acceleration_std_x = nn.Parameter(torch.ones(1) * acceleration_std_x, requires_grad=True)
        self._acceleration_std_y = nn.Parameter(torch.ones(1) * acceleration_std_y, requires_grad=True)

        coef_G = torch.randn(self._state_size, self._state_size, requires_grad=True)
        self._coef_G = nn.Parameter(coef_G, requires_grad=True)

        _GR = torch.randn((2, 2)) * 1e-1
        self._GR = nn.Parameter(_GR)

        self._Id = nn.Parameter(torch.eye(self._state_size), requires_grad=False)

        # Transition matrix that defines evolution of position over a time step for a given state
        self._F = nn.Parameter(torch.eye(self._state_size), requires_grad=False)
        self._F[0, 2] = args.dt
        self._F[1, 3] = args.dt

        # Command matrix that defines how commands modify the state
        self._n_command = 2
        self._B = nn.Parameter(torch.zeros(self._state_size, self._n_command, requires_grad=False), requires_grad=False) # actions are accelerations
        self._B[0, 0] = 0#args.dt * args.dt / 2
        self._B[1, 1] = 0#args.dt * args.dt / 2
        self._B[2, 0] = args.dt
        self._B[3, 1] = args.dt

    def _init_static(self, batch_size):
        self._Q = self._init_Q(batch_size)

    def _get_jacobian(self, X):
        return self._F.unsqueeze(0).repeat(X.shape[0], 1, 1)

    def _get_Q(self, X, Q_corr):
        # type: (Tensor) -> Tensor
        Q = self._Q.clone().unsqueeze(0).repeat((X.shape[0], 1, 1))
        ax2 = self._acceleration_std_x * self._acceleration_std_x
        ay2 = self._acceleration_std_y * self._acceleration_std_y
        axay = self._acceleration_std_x * self._acceleration_std_y

        submat = torch.empty((1, 2, 2), device=X.device)
        submat[0, 0, 0] = ax2
        submat[0, 1, 1] = ay2
        submat[0, 0, 1] = axay
        submat[0, 1, 0] = axay

        submat = submat.repeat((Q.shape[0], 1, 1))

        Q[:, :2, :2] *= submat
        Q[:, 2:, 2:] *= submat
        Q[:, :2, 2:] *= submat
        Q[:, 2:, :2] *= submat

        if Q_corr is not None:
            Q = Q_corr.transpose(2, 1) @ Q @ Q_corr

        return Q

    def _get_R(self):
        R = torch.matmul(self._GR, self._GR.transpose(1, 0))
        return R

    def _pred_state(self, X):
        return torch.matmul(self._F, X)

    def _init_Q(self, batch_size):
        Rho = torch.matmul(self._coef_G, self._coef_G.transpose(1, 0))
        G = torch.zeros(self._state_size, requires_grad=False, device=self._H.device)
        G[0:2] = self._dt * self._dt / 2
        G[2:4] = self._dt
        Q = torch.matmul(G.unsqueeze(1), G.unsqueeze(0)) * Rho
        return Q

    def _init_P(self, batch_size):
        # type: (int) -> Tensor
        R = torch.zeros((2, 2), device=self._H.device)
        R[0, 0] = self._position_std_x ** 2
        R[1, 1] = self._position_std_y ** 2
        P = torch.zeros((batch_size, self._state_size, self._state_size), device=self._H.device)
        P += self._H_inv @ R @ self._H
        P[:, 2, 2] = self._velocity_std_x * self._velocity_std_x
        P[:, 3, 3] = self._velocity_std_x * self._velocity_std_y
        return P

    def _apply_command(self, X, command):
        u = command[:, :2]
        X_corr = torch.matmul(self._B, u.unsqueeze(2))
        return X_corr, command[:, 2:].view(-1, self._state_size, self._state_size)