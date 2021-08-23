import timeit

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from utils import get_whitenoise, get_spectrogram


def main():
    repeat = 5
    number = 10

    sample_rate = 16000
    n_fft = 400
    win_length = n_fft
    hop_length = n_fft // 4
    window = torch.hann_window(win_length)
    power = 1
    duration = 20
    n_iter = 8
    momentum = 0.99

    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=2)
    specgram = get_spectrogram(waveform, n_fft=n_fft, hop_length=hop_length, power=power,
                               win_length=win_length, window=window)
    specgram_np = specgram[0].numpy()
    length = waveform.size(1)

    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                input = torch.clone(specgram).detach().to(device, dtype)
                window = torch.clone(window).detach().to(device, dtype)

                transform_fn = torchaudio.functional.griffinlim
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)

                fn_str = 'transform_fn(input, window=window, n_fft=n_fft, hop_length=hop_length, win_length=win_length, ' \
                                      'power=power, n_iter=n_iter, momentum=momentum, length=length, rand_init=False)'

                # To avoid the first cuda run being slow
                # https://forums.developer.nvidia.com/t/execution-time-the-first-execution-time-is-always-slow/2387
                exec(fn_str)

                res = timeit.repeat(fn_str, repeat=repeat, number=number,
                                    globals={"transform_fn": transform_fn, "input": input, "window": window, "n_fft": n_fft,
                                             "hop_length": hop_length, "win_length": win_length, "power": power, "n_iter": n_iter,
                                             "momentum": momentum, "length": length})
                print(f"{np.mean(res)} +- {sem(res)}")

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")

        fn_str = "transform_fn(input, n_iter=n_iter, hop_length=hop_length, momentum=momentum, init=None, length=length)"
        transform_fn = librosa.griffinlim
        input = specgram_np.astype(dtype, copy=True)

        exec(fn_str)  # for fair comparison

        res = timeit.repeat(fn_str, repeat=repeat, number=number,
                            globals={"transform_fn": transform_fn, "input": input, "n_iter": n_iter, "hop_length": hop_length,
                                     "momentum": momentum, "length": length})
        print(f"{np.mean(res)} +- {sem(res)}")


if __name__ == "__main__":
    main()