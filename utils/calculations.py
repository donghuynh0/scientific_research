import numpy as np
from numpy import log as ln
from decimal import Decimal, getcontext


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


def pressure(p, T):
    # Set high precision for Decimal calculations
    getcontext().prec = 50

    # Convert inputs to Decimals
    p = Decimal(str(p))
    T = Decimal(str(T))

    # Helper for summation
    def Sum(start, end, f):
        return sum(f(i) for i in range(start, end + 1))

    # Exponential function using numpy (float)
    exp = lambda x: Decimal(str(np.exp(float(x))))

    # Coefficients
    x = [
        Decimal(0),
        Decimal("0.8623085097507421"),
        Decimal("2.976218765822098"),
        Decimal("-8.402230115796038"),
        Decimal("0.1054136629203555"),
        Decimal("-0.8564583828174598"),
        Decimal("1.582759470107601"),
        Decimal("0.7639421948305453"),
        Decimal("1.753173414312048"),
        Decimal("2.798291772190376e+03"),
        Decimal("-4.8394220260857657e-02"),
        Decimal("0.9963265197721935"),
        Decimal("-3.698000291272493e+01"),
        Decimal("2.084012299434647e+01"),
        Decimal("8.305402124717285e+01"),
        Decimal("-9.574799715203068e+02"),
        Decimal("-1.477746229234994e+02"),
        Decimal("6.398607852471505e+01"),
        Decimal("1.603993673294834e+01"),
        Decimal("6.805916615864377e+01"),
        Decimal("-2.791293578795945e+03"),
        Decimal("-6.245128304568454"),
        Decimal("-8.116836104958410e+03"),
        Decimal("1.488735559561229e+01"),
        Decimal("-1.059346754655084e+04"),
        Decimal("-1.131607632802822e+02"),
        Decimal("-8.867771540418822e+03"),
        Decimal("-3.986982844450543e+01"),
        Decimal("-4.689270299917261e+03"),
        Decimal("2.593535277438717e+02"),
        Decimal("-2.694523589434903e+03"),
        Decimal("-7.218487631550215e+02"),
        Decimal("1.721802063863269e+02")
    ]

    y = Decimal(3)

    # Coefficients a[i] and b[i] based on x and T
    a = [
        Decimal(0),
        x[1] * T + x[2] * T.sqrt() + x[3] + x[4] / T + x[5] / (T ** 2),
        x[6] * T + x[7] + x[8] / T + x[9] / (T ** 2),
        x[10] * T + x[11] + x[12] / T,
        x[13],
        x[14] / T + x[15] / (T ** 2),
        x[16] / T,
        x[17] / T + x[18] / (T ** 2),
        x[19] / (T ** 2)
    ]

    b = [
        Decimal(0),
        x[20] / (T ** 2) + x[21] / (T ** 3),
        x[22] / (T ** 2) + x[23] / (T ** 4),
        x[24] / (T ** 2) + x[25] / (T ** 3),
        x[26] / (T ** 2) + x[27] / (T ** 4),
        x[28] / (T ** 2) + x[29] / (T ** 3),
        x[30] / (T ** 2) + x[31] / (T ** 3) + x[32] / (T ** 4)
    ]

    # Exponential decay term
    F = exp(-y * p ** 2)

    # Pressure equation
    P = (p * T +
         Sum(1, 8, lambda i: a[i] * p ** (i + 1)) +
         F * Sum(1, 6, lambda i: b[i] * p ** (2 * i + 1)))

    return P


def energy(p, T):
    p = Decimal(str(p))
    T = Decimal(str(T))

    getcontext().prec = 50

    y = Decimal(3)
    x = [
        Decimal(0),
        Decimal("0.8623085097507421"),
        Decimal("2.976218765822098"),
        Decimal("-8.402230115796038"),
        Decimal("0.1054136629203555"),
        Decimal("-0.8564583828174598"),
        Decimal("1.582759470107601"),
        Decimal("0.7639421948305453"),
        Decimal("1.753173414312048"),
        Decimal("2.798291772190376e+03"),
        Decimal("-4.8394220260857657e-02"),
        Decimal("0.9963265197721935"),
        Decimal("-3.698000291272493e+01"),
        Decimal("2.084012299434647e+01"),
        Decimal("8.305402124717285e+01"),
        Decimal("-9.574799715203068e+02"),
        Decimal("-1.477746229234994e+02"),
        Decimal("6.398607852471505e+01"),
        Decimal("1.603993673294834e+01"),
        Decimal("6.805916615864377e+01"),
        Decimal("-2.791293578795945e+03"),
        Decimal("-6.245128304568454"),
        Decimal("-8.116836104958410e+03"),
        Decimal("1.488735559561229e+01"),
        Decimal("-1.059346754655084e+04"),
        Decimal("-1.131607632802822e+02"),
        Decimal("-8.867771540418822e+03"),
        Decimal("-3.986982844450543e+01"),
        Decimal("-4.689270299917261e+03"),
        Decimal("2.593535277438717e+02"),
        Decimal("-2.694523589434903e+03"),
        Decimal("-7.218487631550215e+02"),
        Decimal("1.721802063863269e+02")
    ]

    Sum = lambda i, max, f: np.sum( [f(i) for i in range(i, max+1)] )

    ln = lambda x: x.ln()
    exp = lambda x: np.exp(x)
    F = exp(-y * p ** 2)

    c = [
        0,
        x[2] * (T**Decimal(1/2)) / 2 + x[3] + 2 * x[4] / T + 3 * x[5] / T ** 2,
        x[7] + 2 * x[8] / T + 3 * x[9] / T ** 2,
        x[11] + 2 * x[12] / T,
        x[13],
        2 * x[14] / T + 3 * x[15] / T ** 2,
        2 * x[16] / T,
        2 * x[17] / T + 3 * x[18] / T ** 2,
        3 * x[19] / T ** 2
    ]

    d = [
        0,
        3 * x[20] / T ** 2 + 4 * x[21] / T ** 3,  # d_1
        3 * x[22] / T ** 2 + 5 * x[23] / T ** 4,  # d_2
        3 * x[24] / T ** 2 + 4 * x[25] / T ** 3,  # d_3
        3 * x[26] / T ** 2 + 5 * x[27] / T ** 4,  # d_4
        3 * x[28] / T ** 2 + 4 * x[29] / T ** 3,  # d_5
        3 * x[30] / T ** 2 + 4 * x[31] / T ** 3 + 5 * x[32] / T ** 4  # d_6
    ]

    G = [0]*7
    G[1] = (1-F)/(2*y)
    G[2] = -(F*p**2 - 2*G[1])/(2*y)
    G[3] = -(F*p**4 - 4*G[2]) / (2 * y)
    G[4] = -(F*p**6 - 6*G[3]) / (2 * y)
    G[5] = -(F*p**8 - 8*G[4]) / (2 * y)
    G[6] = -(F*p**10 - 10*G[5]) / (2 * y)

    U = Sum(1, 8, lambda i: c[i]*p**i / i) + Sum(1, 6, lambda i: d[i]*G[i])
    return U


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
