import numpy as np

def fft(N, M, f_in, h_f):
    # дополняем нулями
    f_in_pad = np.pad(f_in, ((M - N) // 2,), 'constant', constant_values=(0,))

    # меняем местами
    f_in_pad = np.concatenate((f_in_pad[M // 2:], f_in_pad[:M // 2]), axis=None)

    # преобразуем
    f_out = np.fft.fft(f_in_pad) * h_f

    # меняем местами
    f_out = np.concatenate((f_out[M // 2:], f_out[:M // 2]), axis=None)

    # вырежем центр
    f_out = f_out[M // 2 - N // 2: M // 2 + N // 2]
    return f_out
