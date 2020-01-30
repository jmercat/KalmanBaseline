import torch
import torch.nn as nn
from kalman_basis import KalmanBasis

class CA_model(KalmanBasis):

    def __init__(self, args):
        KalmanBasis.__init__(self, 6, 2, args)
        self.dt = args.dt

        self._position_std_x = nn.Parameter(torch.ones(1) * 5)
        self._position_std_y = nn.Parameter(torch.ones(1) * 5)
        self._velocity_std_x = nn.Parameter(torch.ones(1) * 10)
        self._velocity_std_y = nn.Parameter(torch.ones(1) * 10)
        self._acceleration_std_x = nn.Parameter(torch.ones(1) * 5)
        self._acceleration_std_y = nn.Parameter(torch.ones(1) * 5)
        self._jerk_std_x = nn.Parameter(torch.ones(1) * 3)
        self._jerk_std_y = nn.Parameter(torch.ones(1) * 3)
        self._n_command = 2

        coef_G = torch.randn(self._state_size, self._state_size)
        self._coef_G = nn.Parameter(coef_G)

        GR = torch.randn(2, 2)
        self._GR = nn.Parameter(GR)

        self._B = nn.Parameter(torch.zeros(self._state_size, self._n_command, requires_grad=False), requires_grad=False) # actions are accelerations
        self._F = nn.Parameter(torch.eye(self._state_size, requires_grad=False), requires_grad=False)
        self.Id = nn.Parameter(torch.eye(self._state_size, requires_grad=False), requires_grad=False)

        dt = args.dt

        # Transition matrix that defines evolution of position over a time step for a given state
        self._F[0, 2] = dt
        self._F[0, 4] = dt * dt / 2
        self._F[1, 3] = dt
        self._F[1, 5] = dt * dt / 2
        self._F[2, 4] = dt
        self._F[3, 5] = dt

        # Command matrix that defines how commands modify the state
        self._B[0, 0] = dt * dt * dt / 6
        self._B[1, 0] = dt * dt / 2
        self._B[2, 0] = dt
        self._B[3, 1] = dt * dt * dt / 6
        self._B[4, 1] = dt * dt / 2
        self._B[5, 1] = dt

    def _init_static(self, batch_size):
        self._Q = self._init_Q(batch_size)

    def _get_jacobian(self, X):
        return self._F.unsqueeze(0).repeat(X.shape[0], 1, 1)

    def _get_Q(self, X, Q_corr):
        Q = self._Q.clone().unsqueeze(0).repeat((X.shape[0], 1, 1))
        jx2 = self._jerk_std_x * self._jerk_std_x
        jy2 = self._jerk_std_y * self._jerk_std_y
        jxy = self._jerk_std_x * self._jerk_std_y

        submat = torch.empty((1, 2, 2), device=X.device)
        submat[0, 0, 0] = jx2
        submat[0, 1, 1] = jy2
        submat[0, 0, 1] = jxy
        submat[0, 1, 0] = jxy

        Q *= submat.repeat((Q.shape[0], 3, 3))

        if Q_corr is not None:
            Q *= Q_corr + 1

        return Q

    def _init_Q(self, batch_size):
        Rho = torch.matmul(self._coef_G, self._coef_G.transpose(1, 0))
        G = torch.zeros(self._state_size, requires_grad=False, device=self._H.device)
        G[0:2] = self._dt * self._dt * self._dt / 6
        G[2:4] = self._dt * self._dt / 2
        G[4:6] = self._dt
        Q = torch.matmul(G.unsqueeze(1), G.unsqueeze(0)) * Rho
        return Q

    def _get_R(self):
        return torch.matmul(self._GR, self._GR.transpose(1, 0))

    def _pred_state(self, X):
        return torch.matmul(self._F, X)

    def _init_P(self, batch_size):
        R = torch.zeros((2, 2), device=self._H.device)
        R[0, 0] = self._position_std_x ** 2
        R[1, 1] = self._position_std_y ** 2
        P = torch.zeros((batch_size, self._state_size, self._state_size), device=self._H.device)
        P += self._H_inv @ R @ self._H
        P[:, 2, 2] = self._velocity_std_x * self._velocity_std_x
        P[:, 3, 3] = self._velocity_std_x * self._velocity_std_y
        P[:, 4, 4] = self._acceleration_std_x * self._acceleration_std_x
        P[:, 5, 5] = self._acceleration_std_y * self._acceleration_std_y
        return P

    def _apply_command(self, X, command):
        u = command[:, :2]
        X_corr = torch.matmul(self._B, u.unsqueeze(2))
        return X_corr, command[:, 2:].view(-1, self._state_size, self._state_size)