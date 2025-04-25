import numpy as np
from numpy import log as ln


# calculate D*rho at rho is zero
def dpt(T):
    a11 = np.array([0, 0.125431, -0.167256, -0.265865, 1.59760, -1.19088, 0.264833])
    a22 = np.array([0, 0.310810, -0.171211, -0.715805, 2.48678, -1.78317, 0.394405])

    w11 = lambda T: (-1 / 6) * ln(T) + np.sum(
        [a11[i] * T ** (-(i - 1) / 2) for i in range(1, 7)]
    )
    w22 = lambda T: (-1 / 6) * ln(T) + ln(17 / 18) + np.sum(
        [a22[i] * T ** (-(i - 1) / 2) for i in range(1, 7)]
    )

    dw11_dt = lambda T: -(1 / (6 * T)) + np.sum(
        [a11[i] * (-(i - 1) / 2) * T ** ((-i - 1) / 2) for i in range(1, 7)]
    )

    dw11_d2t = lambda T: (1 / (6 * T ** 2)) + np.sum(
        [a11[i] * (-(i - 1) / 2) * ((-i - 1) / 2) * T ** ((-i - 3) / 2) for i in range(1, 7)]
    )

    omega11 = lambda T: np.exp(w11(T))
    omega22 = lambda T: np.exp(w22(T))

    omega12 = lambda T: omega11(T) + 1 / (1 + 2) * T * dw11_dt(T)

    dw12_dt = lambda T: dw11_dt(T) + 1 / (1 + 2) * (dw11_dt(T) + T * dw11_d2t(T))

    omega13 = lambda T: omega12(T) + 1 / (2 + 2) * T * (dw12_dt(T))

    A = lambda T: omega22(T) / omega11(T)
    B = lambda T: (5 * omega12(T) - 4 * omega13(T)) / omega11(T)
    C = lambda T: omega12(T) / omega11(T)

    Delta = lambda T: (6 * C(T) - 5) ** 2 / (55 - 12 * B(T) + 16 * A(T))

    fdp = lambda T: 1 / (1 - Delta(T))
    return (3 / 8) * np.sqrt(T / np.pi) * (fdp(T)) / (omega11(T))


"""calculate loss functions"""


# Without weight
def mse_no_weight(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae_no_weight(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def relative_error_no_weight(y_true, y_pred):
    relative_errors = np.abs(y_true - y_pred) / np.where(y_true != 0, y_true, 1)
    return np.mean(relative_errors) * 100


# With weight (8)
def mse_weighted(y_true, y_pred, rho):
    weights = np.where(rho == 0, 8, 1)
    return np.mean(weights * (y_true - y_pred) ** 2)


def mae_weighted(y_true, y_pred, rho):
    weights = np.where(rho == 0, 8, 1)
    return np.mean(weights * np.abs(y_true - y_pred))


def relative_error_weighted(y_true, y_pred, rho):
    relative_errors = np.abs(y_true - y_pred) / np.where(y_true != 0, y_true, 1)
    weights = np.where(rho == 0, 8, 1)
    return np.mean(weights * relative_errors) * 100
