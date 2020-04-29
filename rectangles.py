import numpy as np


def rectangles(N, h_f, x_in, f_in, x_out):
    x_2d = np.broadcast_to(x_in[:, np.newaxis], (N, N))
    u_2d = np.broadcast_to(x_out[np.newaxis, :], (N, N))
    A = np.exp((-2 * np.pi * 1j) * x_2d * u_2d)
    A = A * np.broadcast_to(f_in[:, np.newaxis], (N, N))

    int_weights = np.ones(N)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= h_f

    # scale rows by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis], (N, N))

    result = np.sum(A, axis=0)

    return result

def rectangles_2d(N, h_f, x_in, y_in, f_in, x_out):
    shape = (N, N, N, N)

    # first dimension - x
    x_4d = np.broadcast_to(x_in[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # second dimension - y
    y_4d = np.broadcast_to(y_in[np.newaxis, :, np.newaxis, np.newaxis], shape)

    # third dimension - u
    u_4d = np.broadcast_to(x_out[np.newaxis, np.newaxis, :, np.newaxis], shape)
    # forth dimension - v
    v_4d = np.broadcast_to(x_out[np.newaxis, np.newaxis, np.newaxis, :], shape)

    # exp values
    A = np.exp((-2 * np.pi * 1j) * (x_4d * u_4d + y_4d * v_4d))

    # scale d1 and d2 by f(x, y)
    A = A * np.broadcast_to(f_in[:, :, np.newaxis, np.newaxis], shape)

    int_weights = np.ones(N)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= h_f

    # scale d1 by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # scale d2 by int_weights
    A = A * np.broadcast_to(int_weights[np.newaxis, :, np.newaxis, np.newaxis], shape)

    result = A
    result = np.sum(result, axis=0)
    result = np.sum(result, axis=0)

    return result
