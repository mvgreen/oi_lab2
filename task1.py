# Реализовать одномерное финитное преобразование фурье через бпф
import numpy as np
import matplotlib.pyplot as mplot
import rectangles
import fft

def plot_complex(x, y, title):
    x_abs, = mplot.plot(x, np.abs(y))
    x_abs.set_label('amplitude')
    mplot.title(title)
    mplot.legend()
    mplot.show()

    x_ph, = mplot.plot(x, np.angle(y))
    x_ph.set_label('phase')
    mplot.title(title)
    mplot.legend()
    mplot.show()

def plot_complex_comparison(x, y1, y2, title):
    a1_abs, = mplot.plot(x, np.abs(y1))
    a1_abs.set_label('integral method amplitude')
    a2_abs, = mplot.plot(x, np.abs(y2))
    a2_abs.set_label('fft method amplitude')
    mplot.title(title)
    mplot.legend()
    mplot.show()

    p1_abs, = mplot.plot(x, np.angle(y1))
    p1_abs.set_label('integral method phase')
    p2_abs, = mplot.plot(x, np.angle(y2))
    p2_abs.set_label('fft method phase')
    mplot.title(title)
    mplot.legend()
    mplot.show()

    ae1_abs, = mplot.plot(x, np.abs(np.abs(y1) - np.abs(y2)))
    ae1_abs.set_label('amplitude error')
    mplot.title(title)
    mplot.legend()
    mplot.show()

    pe1_abs, = mplot.plot(x, np.abs(np.angle(y1) - np.angle(y2)))
    pe1_abs.set_label('phase error')
    mplot.title(title)
    mplot.legend()
    mplot.show()


def f_gauss(x):
    return np.exp(-(np.power(x, 2)))


def task1(N, M, h_f, x_in, x_out, f_in):

    plot_complex(x_in, f_in, 'f(x)')

    # интегральное решение
    integral_out = rectangles.rectangles(N, h_f, x_in, f_in, x_out)
    plot_complex(x_out, integral_out, 'integral')

    # через бпф
    f_out = fft.fft(N, M, f_in, h_f)
    plot_complex(x_out, f_out, 'F(x)')

    # сравнение
    plot_complex_comparison(x_out, integral_out, f_out, 'comparison')


# Размеры массивов, N должно быть четным, M - степенью двойки
N = 1 << 10
M = 1 << 15

a = 4
h_f = a * 2 / (N - 1)
b = N ** 2 / (4 * a * M)
h_out = b * 2 / (N - 1)

x_in = np.array(np.linspace(-a, a, N), dtype=np.complex)
x_out = np.array(np.linspace(-b, b, N), dtype=np.complex)

#f_in = f_gauss(x_in)
#task1(N, M, h_f, x_in, x_out, f_in)

f_in = np.power(x_in, 2)
task1(N, M, h_f, x_in, x_out, f_in)
