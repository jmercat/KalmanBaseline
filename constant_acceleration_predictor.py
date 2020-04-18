import torch
import torch.nn as nn
from kalman_basis import KalmanBasis

class CA_model(KalmanBasis):

    def __init__(self, args):
        KalmanBasis.__init__(self, 6, 2, args)
        self.dt = args.dt

        self._position_std_x = nn.Parameter(torch.ones(1) * 0.1)
        self._position_std_y = nn.Parameter(torch.ones(1) * 3.2)
        self._velocity_std_x = nn.Parameter(torch.ones(1) * 0.2)
        self._velocity_std_y = nn.Parameter(torch.ones(1) * 8)
        self._acceleration_std_x = nn.Parameter(torch.ones(1) * 1)
        self._acceleration_std_y = nn.Parameter(torch.ones(1) * 2.7)
        self._jerk_std_x = nn.Parameter(torch.ones(1) * 1)
        self._jerk_std_y = nn.Parameter(torch.ones(1) * 1)
        self._n_command = 2

        coef_G = torch.randn(self._state_size)
        self._coef_G = nn.Parameter(coef_G)

        GR = torch.randn(2, 2)
        self._GR = nn.Parameter(GR)

        self.Id = nn.Parameter(torch.eye(self._state_size), requires_grad=False)

        dt = args.dt

        # Transition matrix that defines evolution of position over a time step for a given state
        self._F = nn.Parameter(torch.eye(self._state_size), requires_grad=False)
        self._F[0, 2] = dt
        self._F[0, 4] = dt * dt / 2
        self._F[1, 3] = dt
        self._F[1, 5] = dt * dt / 2
        self._F[2, 4] = dt
        self._F[3, 5] = dt

        # Command matrix that defines how commands modify the state
        self._B = nn.Parameter(torch.zeros(self._state_size, self._n_command), requires_grad=False) # actions are accelerations
        self._B[0, 0] = 0#dt * dt * dt / 6
        self._B[1, 1] = 0#dt * dt * dt / 6
        self._B[2, 0] = 0#dt * dt / 2
        self._B[3, 1] = 0#dt * dt / 2
        self._B[4, 0] = dt
        self._B[5, 1] = dt

    def _get_jacobian(self, X):
        return self._F.unsqueeze(0).repeat(X.shape[0], 1, 1)

    def _get_Q(self, X, Q_corr):
        Q = self._init_Q().clone().unsqueeze(0).repeat((X.shape[0], 1, 1))
        jx2 = self._jerk_std_x * self._jerk_std_x
        jy2 = self._jerk_std_y * self._jerk_std_y
        jxy = self._jerk_std_x * self._jerk_std_y

        submat = torch.zeros((1, self._state_size, self._state_size), device=X.device)
        submat[0, 0, 0] = jx2
        submat[0, 1, 1] = jy2
        submat[0, 2, 2] = jx2
        submat[0, 3, 3] = jy2
        submat[0, 4, 4] = jx2
        submat[0, 5, 5] = jy2

        if Q_corr is not None:
            # Q = Q_corr.transpose(2, 1) @ Q @ Q_corr
            Q = Q_corr
        else:
            Q = torch.matmul(Q, submat)

        return Q

    def _init_Q(self):
        G = torch.zeros((self._state_size, 2), requires_grad=False, device=self._H.device)
        G[0, 0] = self._dt * self._dt * self._dt / 6
        G[1, 1] = self._dt * self._dt * self._dt / 6
        G[2, 0] = self._dt * self._dt / 2
        G[3, 1] = self._dt * self._dt / 2
        G[4, 0] = self._dt
        G[5, 1] = self._dt
        Q = torch.matmul((G * torch.tanh(self._coef_G).unsqueeze(1)),
                         (G * torch.tanh(self._coef_G).unsqueeze(1)).transpose(1, 0))
        return Q

    def _get_R(self):
        return torch.matmul(self._GR, self._GR.transpose(1, 0))

    def _pred_state(self, X):
        return torch.matmul(self._F, X)

    def _init_P(self, batch_size):
        P = torch.zeros((batch_size, self._state_size, self._state_size), device=self._H.device)
        P[:, 0, 0] = self._position_std_x * self._position_std_x
        P[:, 1, 1] = self._position_std_y * self._position_std_y
        P[:, 2, 2] = self._velocity_std_x * self._velocity_std_x
        P[:, 3, 3] = self._velocity_std_x * self._velocity_std_y
        P[:, 4, 4] = self._acceleration_std_x * self._acceleration_std_x
        P[:, 5, 5] = self._acceleration_std_y * self._acceleration_std_y
        return P

    def _init_X(self, Z):
        V = (Z[1] - Z[0]).clone()/self._dt
        A = (Z[2] + Z[0] - 2*Z[1]).clone()/(self._dt * self._dt)

        X = torch.zeros((Z.shape[1], self._state_size), device=Z.device)
        X[:, 0] = Z[0, :, 0]
        X[:, 1] = Z[0, :, 1]
        X[:, 2] = V[:, 0]
        X[:, 3] = V[:, 1]
        X[:, 4] = A[:, 0]
        X[:, 5] = A[:, 1]
        return X[:, :, None]

    def _apply_command(self, X, command):
        u = command[:, :2]
        X_corr = torch.matmul(self._B, u.unsqueeze(2))
        return X_corr, command[:, 2:].view(-1, self._state_size, self._state_size)
