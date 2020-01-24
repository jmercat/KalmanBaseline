import torch
import torch.nn as nn
import numpy as np

from kalman_basis import KalmanBasis


class Bicycle_model(KalmanBasis):
    """
    This class implements kalman_basis for the bicycle model with a state [x, y, theta, v, a, w]
    which are position, yaw (heading angle), velocity, acceleration, wheel_angle
    """
    def __init__(self, dt,
                 velocity_std=10, yaw_std=15, pos_std=1,
                 jerk=5, wheel_jerk=3, a_std=1, wheel_std=3):
        # type: (float, float, float, float, float, float, float, float) -> None
        KalmanBasis.__init__(self, 6, 3, dt)

        self._vehicle_length = nn.Parameter(torch.ones(1) * 2.6, requires_grad=False)
        self._velocity_std = nn.Parameter(torch.ones(1) * velocity_std, requires_grad=True)
        self._yaw_std = nn.Parameter(torch.ones(1) * yaw_std * np.pi / 180, requires_grad=True)
        self._pos_std = nn.Parameter(torch.ones(1) * pos_std, requires_grad=True)
        self._jerk = nn.Parameter(torch.ones(1) * jerk, requires_grad=True)
        self._a_std = nn.Parameter(torch.ones(1) * a_std, requires_grad=True)
        self._wheel_jerk = nn.Parameter(torch.ones(1) * wheel_jerk * np.pi / 180, requires_grad=True)
        self._wheel_std = nn.Parameter(torch.ones(1) * wheel_std * np.pi / 180, requires_grad=True)

        coef_G = torch.randn(self._state_size, self._state_size)*0.1 + torch.eye(self._state_size)
        self._coef_G = nn.Parameter(coef_G, requires_grad=True)

        _GR = torch.randn((3, 3))*0.001
        self._GR = nn.Parameter(_GR)

    def _init_static(self, batch_size):
        self._Q = self._init_Q(batch_size)
        A = torch.eye(self._state_size, requires_grad=False, device=self._H.device)
        self._J = A.unsqueeze(0).repeat([batch_size, 1, 1])

    def _get_jacobian(self, X):
        # type: (Tensor) -> Tensor
        theta = X[:, 2, 0].clone()
        wheel_angle = X[:, 5, 0].clone()
        velocity = X[:, 3, 0].clone()
        acc = X[:, 4, 0].clone()
        tan_w = torch.tan(wheel_angle)
        curvature = tan_w / self._vehicle_length
        dt = self._dt
        theta_05 = theta + curvature * (velocity + acc * dt / 2)
        velocity_05 = velocity + dt * acc / 2

        cos = torch.cos(theta_05)
        sin = torch.sin(theta_05)

        J = self._J.clone()

        J[:, 0, 2] = - dt * velocity_05.clone() * sin.clone()
        J[:, 1, 2] = dt * velocity_05.clone() * cos.clone()

        J[:, 2, 3] = dt * curvature.clone()
        J[:, 0, 3] = dt * cos.clone() - dt / 2 * velocity_05.clone() * sin.clone() * J[:, 2, 3].clone()
        J[:, 1, 3] = dt * sin.clone() + dt / 2 * velocity_05.clone() * cos.clone() * J[:, 2, 3].clone()

        J[:, 2, 4] = dt * dt / 2 * curvature.clone()
        J[:, 0, 4] = dt * dt / 2 * cos.clone() - dt / 2 * velocity_05.clone() * sin.clone() * J[:, 2, 4].clone()
        J[:, 1, 4] = dt * dt / 2 * sin.clone() + dt / 2 * velocity_05.clone() * cos.clone() * J[:, 2, 4].clone()
        J[:, 3, 4] = dt

        J[:, 2, 5] = dt * velocity_05.clone() * (1 + tan_w.clone() * tan_w.clone()) / self._vehicle_length
        J[:, 0, 5] = - dt / 2 * velocity_05.clone() * sin.clone() * J[:, 2, 5].clone()
        J[:, 1, 5] = dt / 2 * velocity_05.clone() * cos.clone() * J[:, 2, 5].clone()

        return J

    def _get_Q(self, X):
        dt = self._dt
        angle = X[:, 2].clone()
        Q = torch.zeros((X.shape[0], self._state_size, self._state_size), device=X.device)
        G = torch.from_numpy(np.array([dt ** 3 / 6, dt ** 2 / 2, dt]).astype('float32')).to(X.device)
        GGt = G[:, None] @ G[None, :]
        xva_rows = np.array([0, 3, 4])
        for ei, i in enumerate(xva_rows):
            for ej, j in enumerate(xva_rows):
                Q[:, i, j] = self._jerk ** 2 * GGt[ei, ej].clone()
        yva_rows = np.array([1, 3, 4])
        for ei, i in enumerate(yva_rows):
            for ej, j in enumerate(yva_rows):
                Q[:, i, j] = self._jerk ** 2 * GGt[ei, ej].clone()

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        Q[:, 0] = Q[:, 0].clone() * cos.clone()
        Q[:, :, 0] = Q[:, :, 0].clone() * cos.clone()
        Q[:, 1] = Q[:, 1].clone() * sin.clone()
        Q[:, :, 1] = Q[:, :, 1].clone() * sin.clone()

        G = self._wheel_jerk * torch.from_numpy(np.array([dt ** 2 / (2 * self._vehicle_length), dt]).astype('float32')).to(X.device)
        GGt = G[:, None] @ G[None, :]
        tyaw_rows = np.array([2, 5])
        for ei, i in enumerate(tyaw_rows):
            for ej, j in enumerate(tyaw_rows):
                Q[:, i, j] = GGt[ei, ej].clone()
        return Q

    def _get_R(self):
        R = torch.matmul(self._GR, self._GR.transpose(1, 0))
        return R

    def _pred_state(self, X):
        # type: (Tensor) -> Tensor
        angle = X[:, 2, 0].clone()
        wheel_angle = X[:, 5, 0].clone()
        velocity = X[:, 3, 0].clone()
        acc = X[:, 4, 0].clone()
        tan_w = torch.tan(wheel_angle)
        curvature = tan_w / self._vehicle_length
        dt = self._dt
        yaw_rate = curvature * (velocity + acc * dt / 2)
        X[:, 0, 0] = X[:, 0, 0] + torch.cos(angle + yaw_rate * dt / 2) * dt * (velocity + dt * acc / 2)
        X[:, 1, 0] = X[:, 1, 0] + torch.sin(angle + yaw_rate * dt / 2) * dt * (velocity + dt * acc / 2)
        X[:, 2, 0] = X[:, 2, 0] + yaw_rate * dt
        X[:, 3, 0] = X[:, 3, 0] + dt * acc
        return X

    def _init_Q(self, batch_size):
        device = self._H.device
        dt = self._dt
        Rho = torch.matmul(self._coef_G, self._coef_G.transpose(1, 0))
        G = torch.zeros(self._state_size, requires_grad=False, device=device)
        G[0:2] = dt ** 3 / 6
        G[2:4] = dt ** 2 / 2
        G[4:6] = dt
        Q = G.unsqueeze(1) @ G.unsqueeze(0) * Rho
        return Q.unsqueeze(0).repeat(batch_size, 1, 1)

    def _init_P(self, batch_size):
        # type: (int) -> Tensor
        P = torch.zeros((batch_size, self._state_size, self._state_size), device=self._H.device)
        P[:, 0, 0] = self._pos_std ** 2
        P[:, 1, 1] = self._pos_std ** 2
        P[:, 2, 2] = self._yaw_std ** 2
        P[:, 3, 3] = self._velocity_std ** 2
        P[:, 4, 4] = self._a_std ** 2
        P[:, 5, 5] = self._wheel_std ** 2
        return P
