import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import rectangles
import fft


def plot_2d(x, y, z):
    plt.imshow(np.abs(z), extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()

    plt.imshow(np.angle(z), extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()


def plot_2d_compare(x, y, f1, f2):
    plt.imshow(np.abs(np.abs(f1) - np.abs(f2)), extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()

    plt.imshow(np.abs(np.angle(f1)-np.angle(f2)), extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()


    x_sqr = np.broadcast_to(x[:, np.newaxis], (N, N))
    y_sqr = np.broadcast_to(y[np.newaxis, :], (N, N))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_sqr, y_sqr, np.abs(np.abs(f1) - np.abs(f2)), cmap=cm.coolwarm)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_sqr, y_sqr, np.abs(np.angle(f1)-np.angle(f2)), cmap=cm.coolwarm)
    plt.show()


def f_gauss(x, y):
    return np.exp(-(np.power(x, 2))-(np.power(y, 2)))


def f_custom(x, y):
    return np.power(x, 2) * np.power(y, 2)


def task2(N, h_f, x_in, y_in, x_out, y_out, f_in):
    plot_2d(x_in, y_in, f_in)

    # интеграл
    integral = rectangles.rectangles_2d(N, h_f, x_in, y_in, f_in, x_out)
    plot_2d(x_out, y_out, integral)

    # бпф
    f_out = np.zeros((N, N))

    # применяем сначала построчно
    for line in range(0, N):
        f_out[line, :] = fft.fft(N, M, f_in[line,:], h_f)

    # затем по столбцам
    for col in range(0, N):
        f_out[:, col] = fft.fft(N, M, f_out[:, col], h_f)

    plot_2d(x_out, y_out, f_out)

    plot_2d_compare(x_out, y_out, integral, f_out)


# Размеры массивов, N должно быть четным, M - степенью двойки
N = 1 << 6
M = 1 << 8

shape = (N, N)

a = 4
h_f = a * 2 / (N - 1)
b = N ** 2 / (4 * a * M)
h_out = b * 2 / (N - 1)

x_in = np.array(np.linspace(-a, a, N), dtype=np.float)
y_in = np.array(np.linspace(-a, a, N), dtype=np.float)

x_out = np.array(np.linspace(-b, b, N), dtype=np.float)
y_out = np.array(np.linspace(-b, b, N), dtype=np.float)

x_in_sqr = np.broadcast_to(x_in[:, np.newaxis], (N, N))
y_in_sqr = np.broadcast_to(y_in[np.newaxis, :], (N, N))

# f_in = f_gauss(x_in_sqr, y_in_sqr)
# task2(N, h_f, x_in, y_in, x_out, y_out, f_in)

f_in = f_custom(x_in_sqr, y_in_sqr)
task2(N, h_f, x_in, y_in, x_out, y_out, f_in)
