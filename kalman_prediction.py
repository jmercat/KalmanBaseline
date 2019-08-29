import torch
import torch.nn as nn


class KalmanCV(nn.Module):

    def __init__(self, dt):
        super(KalmanCV, self).__init__()

        self.dt = dt
        self.n_var = 4

        self.position_std_x = nn.Parameter(torch.ones(1) * 0.12)
        self.position_std_y = nn.Parameter(torch.ones(1) * 15.21)
        self.velocity_std_x = nn.Parameter(torch.ones(1) * 0.014)
        self.velocity_std_y = nn.Parameter(torch.ones(1) * 31.37)
        self.acceleration_std_x = nn.Parameter(torch.ones(1) * 0.53)
        self.acceleration_std_y = nn.Parameter(torch.ones(1) * 4.33)
        self.GR_ = torch.tensor([[3.3850e-28, 5.0867e-19],
                                 [-1.2177e-27, 6.0785e-19]])
        # self.GR_ = torch.randn((2, 2)) * 1e-1
        self.GR_ = nn.Parameter(self.GR_)

        if torch.cuda.is_available():
            self.H = torch.zeros(2, self.n_var, requires_grad=False).cuda() # observations are x, y
            self.H_t = torch.zeros(self.n_var, 2, requires_grad=False).cuda()  # observations are x, y
            self.F = torch.eye(self.n_var, requires_grad=False).cuda()
            self.G_ = torch.zeros(self.n_var, requires_grad=False).cuda()
            self.Id = torch.eye(self.n_var, requires_grad=False).cuda()
        else:
            self.H = torch.zeros(2, self.n_var, requires_grad=False)  # observations are x, y
            self.H_t = torch.zeros(self.n_var, 2, requires_grad=False)  # observations are x, y
            self.F = torch.eye(self.n_var, requires_grad=False)
            self.G_ = torch.zeros(self.n_var, requires_grad=False)
            self.Id = torch.eye(self.n_var, requires_grad=False)

        self.G_[0] = dt * dt / 2
        self.G_[1] = dt
        self.G_[2] = dt * dt / 2
        self.G_[3] = dt
        # coef_G = torch.randn(self.n_var, self.n_var)
        coef_G = torch.tensor([[-4.0431e-01, -6.2302e-03,  1.9845e+00, -4.9377e+00],
                               [ 1.7677e-02,  8.6025e-05,  3.9515e-01,  4.6597e-02],
                               [-1.0531e+00, -3.6640e-04,  2.8276e-02, -1.0620e-01],
                               [-5.4513e-01,  5.8622e-05, -1.4437e-02,  2.4759e-02]])
        self.coef_G = nn.Parameter(coef_G)

        # Observation matrix that mask speeds and keeps positions
        self.H[0, 0] = 1
        self.H[1, 2] = 1
        self.H_t[0, 0] = 1
        self.H_t[2, 1] = 1

        # Transition matrix that defines evolution of position over a time step for a given state
        self.F[0, 1] = dt
        self.F[2, 3] = dt

    def _compute_hist_filter(self, Z):
        V0 = (Z[1] - Z[0])/self.dt
        X, P = self._kalman_init(Z[0], V0)
        for i in range(1, len(Z)):
            X, P = self._kalman_pred(X, P)
            y, S = self._kalman_innovation(X, Z[i], P)
            X, P = self._kalman_update(X, P, S, y)
        return X, P

    def forward(self, hist, len_pred):
        batch_size = hist.shape[1]
        hist = hist.unsqueeze(3)
        X, P = self._compute_hist_filter(hist)

        pred_mu = []
        pred_P = []
        for i in range(len_pred):
            X, P = self._kalman_pred(X, P)
            temp_X_out = torch.matmul(self.H, X).transpose(2, 1)
            temp_P_out = torch.matmul(torch.matmul(self.H, P), self.H_t)
            pred_mu += [temp_X_out]
            pred_P += [temp_P_out]

        pred_mu = torch.stack(pred_mu)
        pred_P = torch.stack(pred_P)
        pred_P = pred_P.view(len_pred, batch_size, 2, 2)
        pred_mu = pred_mu.view(len_pred, batch_size, 2)
        sigma_x = torch.sqrt(pred_P[:, :, 0, 0].view(len_pred, batch_size, 1))
        sigma_y = torch.sqrt(pred_P[:, :, 1, 1].view(len_pred, batch_size, 1))
        rho = (pred_P[:, :, 0, 1] + pred_P[:, :, 1, 0]).view(len_pred, batch_size, 1)/(2*sigma_x*sigma_y)
        return torch.cat([pred_mu, sigma_x, sigma_y, rho], 2)

    def _kalman_pred(self, X, P):
        X_pred = torch.matmul(self.F, X)

        Rho = torch.matmul(self.coef_G, self.coef_G.transpose(1, 0))
        Q = torch.matmul(self.G_.unsqueeze(1), self.G_.unsqueeze(0)) * Rho
        Q[:2, :2] *= self.acceleration_std_x * self.acceleration_std_x
        Q[2:, 2:] *= self.acceleration_std_y * self.acceleration_std_y
        Q[:2, 2:] *= self.acceleration_std_x * self.acceleration_std_y
        Q[2:, :2] *= self.acceleration_std_x * self.acceleration_std_y

        P_pred = torch.matmul(torch.matmul(self.F, P), self.F.transpose(1, 0)) + Q

        return X_pred, P_pred

    def _kalman_innovation(self, X, Z, P):
        R = torch.matmul(self.GR_, self.GR_.transpose(1, 0))
        y = Z - torch.matmul(self.H, X)
        S = torch.matmul(torch.matmul(self.H, P), self.H_t) + R
        return y, S

    def _kalman_update(self, X, P, S, y):
        R = torch.matmul(self.GR_, self.GR_.transpose(1, 0))
        K, _ = torch.solve(torch.matmul(self.H, P.transpose(2, 1)), S.transpose(2, 1))
        K = K.transpose(2, 1)
        X = X + torch.matmul(K, y)

        # Joseph formula for stability
        ImKH = self.Id.unsqueeze(0) - torch.matmul(K, self.H)
        KRK = torch.matmul(torch.matmul(K, R), K.transpose(2, 1))
        P = torch.matmul(torch.matmul(ImKH, P), ImKH.transpose(2, 1)) + KRK
        return X, P

    def _kalman_init(self, Z, V):
        X = torch.matmul(self.H_t, Z)
        X[:, 1] = V[:, 0]
        X[:, 3] = V[:, 1]

        if torch.cuda.is_available():
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var).cuda()
        else:
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var)

        P[:, 0, 0] = self.position_std_x * self.position_std_x
        P[:, 1, 1] = self.velocity_std_x * self.velocity_std_x
        P[:, 2, 2] = self.position_std_y * self.position_std_y
        P[:, 3, 3] = self.velocity_std_y * self.velocity_std_y

        return X, P


class KalmanLSTM(nn.Module):

    def __init__(self, dt):
        super(KalmanLSTM, self).__init__()

        self.dt = dt
        self.n_var = 6
        self.n_command = 2
        self.feature_size = 128
        self.n_layers = 1

        self.position_std_x = nn.Parameter(torch.ones(1) * 0.51)
        self.position_std_y = nn.Parameter(torch.ones(1) * 6.24)
        self.velocity_std_x = nn.Parameter(torch.ones(1) * 0.0016)
        self.velocity_std_y = nn.Parameter(torch.ones(1) * 19.66)
        self.acceleration_std_x = nn.Parameter(torch.ones(1) * 1e-5)
        self.acceleration_std_y = nn.Parameter(torch.ones(1) * 5.46)
        jerk_std = torch.ones(1, 2)
        jerk_std[0, 0] *= 0.90
        jerk_std[0, 1] *= 12.17
        self.jerk_std = nn.Parameter(jerk_std)

        coef_G = torch.tensor(
            [[5.2294e+00, -2.6389e+00, 1.8389e+00, 4.4642e+00, 8.6723e+00, -5.5184e+00],
             [1.6056e+00, -8.1404e-01, 6.4966e-01, 1.4483e+00, 2.6705e+00, -1.6941e+00],
             [-4.3528e-02, 2.2453e-02, -1.5272e-02, -3.4361e-02, -6.8364e-02, 4.5767e-02],
             [1.9162e-02, -1.2546e-02, -6.6970e-01, -1.2119e+00, 7.2434e-02, -2.0197e-02],
             [6.1042e-03, -3.1672e-03, -2.5424e-01, -5.1976e-01, 2.9036e-02, -6.3444e-03],
             [-4.6360e-03, -2.6940e-03, -5.5129e-03, 3.6231e-02, -5.8934e-03, 5.3780e-03]])
        self.coef_G = nn.Parameter(coef_G)

        self.GR_ = torch.tensor([[0.0004, 0.0004], [0.0003, 0.0002]])
        self.GR_ = nn.Parameter(self.GR_)

        if torch.cuda.is_available():
            self.H = torch.zeros(2, self.n_var, requires_grad=False).cuda() # observations are x, y
            self.H_t = torch.zeros(self.n_var, 2, requires_grad=False).cuda()  # observations are x, y
            self.B = torch.zeros(self.n_var, 2, requires_grad=False).cuda() # actions are accelerations
            self.F = torch.eye(self.n_var, requires_grad=False).cuda()
            self.G_ = torch.zeros(self.n_var, 2, requires_grad=False).cuda()
            self.Id = torch.eye(self.n_var, requires_grad=False).cuda()
        else:
            self.H = torch.zeros(2, self.n_var, requires_grad=False)  # observations are x, y
            self.H_t = torch.zeros(self.n_var, 2, requires_grad=False)  # observations are x, y
            self.B = torch.zeros(self.n_var, 2, requires_grad=False) # actions are accelerations
            self.F = torch.eye(self.n_var, requires_grad=False)
            self.G_ = torch.zeros(self.n_var, 2, requires_grad=False)
            self.Id = torch.eye(self.n_var, requires_grad=False)

        # Process noise
        self.G_[0, 0] = dt * dt * dt / 6
        self.G_[1, 0] = dt * dt / 2
        self.G_[2, 0] = dt
        self.G_[3, 1] = dt * dt * dt / 6
        self.G_[4, 1] = dt * dt / 2
        self.G_[5, 1] = dt

        # Observation matrix that mask speeds and keeps positions
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H_t[0, 0] = 1
        self.H_t[3, 1] = 1

        # Transition matrix that defines evolution of position over a time step for a given state
        self.F[0, 1] = dt
        self.F[0, 2] = dt * dt / 2
        self.F[1, 2] = dt
        self.F[3, 4] = dt
        self.F[3, 5] = dt * dt / 2
        self.F[4, 5] = dt

        # Command matrix that defines how commands modify the state
        self.B[0, 0] = dt * dt * dt / 6
        self.B[1, 0] = dt * dt / 2
        self.B[2, 0] = dt
        self.B[3, 1] = dt * dt * dt / 6
        self.B[4, 1] = dt * dt / 2
        self.B[5, 1] = dt

        self.command_feature = nn.Linear(self.n_var, self.feature_size)
        LSTMcells = []
        for i in range(self.n_layers):
            LSTMcells.append(nn.LSTMCell(self.feature_size,
                                         self.feature_size))
        self.LSTMcells = nn.ModuleList(LSTMcells)

        self.command_out = nn.Linear(self.feature_size, self.n_command * (1 + self.n_command))

    def _init_LSTM(self, batch_size):
        if torch.cuda.is_available():
            hx_list = [torch.zeros(batch_size, self.feature_size).cuda() for i in range(len(self.LSTMcells))]
            cx_list = [torch.zeros(batch_size, self.feature_size).cuda() for i in range(len(self.LSTMcells))]
        else:
            hx_list = [torch.zeros(batch_size, self.feature_size) for i in range(len(self.LSTMcells))]
            cx_list = [torch.zeros(batch_size, self.feature_size) for i in range(len(self.LSTMcells))]
        return hx_list, cx_list

    def _get_command(self, X, state):
        hx_list, cx_list = state
        command = torch.tanh(self.command_feature(X.squeeze(2)))
        for j, cell in enumerate(self.LSTMcells):
            hx_list[j], cx_list[j] = cell(command, (hx_list[j], cx_list[j]))
            command = hx_list[j]
        command = self.command_out(command)
        return command, (hx_list, cx_list)

    def _compute_hist_filter(self, Z):
        batch_size = Z.shape[1]
        state = self._init_LSTM(batch_size)
        V0 = (Z[1] - Z[0])/self.dt
        X, P = self._kalman_init(Z[0], V0)
        for i in range(1, len(Z)):
            command, state = self._get_command(X, state)
            X, P = self._kalman_pred(X, P)
            y, S = self._kalman_innovation(X, Z[i], P)
            X, P = self._kalman_update(X, P, S, y)
        return X, P, state

    def forward(self, hist, len_pred):
        batch_size = hist.shape[1]
        hist = hist.unsqueeze(3)
        X, P, state = self._compute_hist_filter(hist)

        pred_mu = []
        pred_P = []
        for i in range(len_pred):
            command, state = self._get_command(X, state)
            X, P = self._kalman_pred(X, P, command)
            temp_X_out = torch.matmul(self.H, X).transpose(2, 1)
            temp_P_out = torch.matmul(torch.matmul(self.H, P), self.H_t)
            pred_mu += [temp_X_out]
            pred_P += [temp_P_out]

        pred_mu = torch.stack(pred_mu)
        pred_P = torch.stack(pred_P)
        pred_P = pred_P.view(len_pred, batch_size, 2, 2)
        pred_mu = pred_mu.view(len_pred, batch_size, 2)
        sigma_x = torch.sqrt(pred_P[:, :, 0, 0].view(len_pred, batch_size, 1))
        sigma_y = torch.sqrt(pred_P[:, :, 1, 1].view(len_pred, batch_size, 1))
        rho = (pred_P[:, :, 0, 1] + pred_P[:, :, 1, 0]).view(len_pred, batch_size, 1)/(2*sigma_x*sigma_y)
        return torch.cat([pred_mu, sigma_x, sigma_y, rho], 2)

    def _kalman_pred(self, X, P, command=None):
        X_pred = torch.matmul(self.F, X)
        Rho = torch.matmul(self.coef_G, self.coef_G.transpose(1, 0))
        Gs = (self.G_ * self.jerk_std).unsqueeze(0)
        if command is not None:
            u = command[:, :2]
            Gs = torch.matmul(Gs, command[:, 2:].view(command.shape[0], 2, 2))
            X_pred += torch.matmul(self.B, u.unsqueeze(2))
        Q = torch.matmul(Gs, Gs.transpose(2, 1)) * Rho

        P_pred = torch.matmul(torch.matmul(self.F, P), self.F.transpose(1, 0)) + Q

        return X_pred, P_pred

    def _kalman_innovation(self, X, Z, P):
        R = torch.matmul(self.GR_, self.GR_.transpose(1, 0))
        y = Z - torch.matmul(self.H, X)
        S = torch.matmul(torch.matmul(self.H, P), self.H_t) + R
        return y, S

    def _kalman_update(self, X, P, S, y):
        R = torch.matmul(self.GR_, self.GR_.transpose(1, 0))
        K, _ = torch.solve(torch.matmul(self.H, P.transpose(2, 1)), S.transpose(2, 1))
        K = K.transpose(2, 1)
        X = X + torch.matmul(K, y)

        # Classic formula
        # P = P - torch.matmul(torch.matmul(K, self.H), P)

        # Joseph formula for stability
        ImKH = self.Id.unsqueeze(0) - torch.matmul(K, self.H)
        KRK = torch.matmul(torch.matmul(K, R), K.transpose(2, 1))
        P = torch.matmul(torch.matmul(ImKH, P), ImKH.transpose(2, 1)) + KRK
        return X, P

    def _kalman_init(self, Z, V):
        X = torch.matmul(self.H_t, Z)
        X[:, 1] = V[:, 0]
        X[:, 3] = V[:, 1]

        if torch.cuda.is_available():
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var).cuda()
        else:
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var)

        P[:, 0, 0] = self.position_std_x * self.position_std_x
        P[:, 1, 1] = self.velocity_std_x * self.velocity_std_x
        P[:, 2, 2] = self.acceleration_std_x * self.acceleration_std_x
        P[:, 3, 3] = self.position_std_y * self.position_std_y
        P[:, 4, 4] = self.velocity_std_y * self.velocity_std_y
        P[:, 5, 5] = self.acceleration_std_y * self.acceleration_std_y

        return X, P
