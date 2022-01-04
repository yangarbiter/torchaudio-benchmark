import sys
sys.path.append("../")

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from memory_profiler import memory_usage
from utils import get_whitenoise, get_spectrogram, update_results, memusage_kwargs


def main():
    results = {}
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

                def run_fn(input, window, n_fft, hop_length, win_length, power, n_iter, momentum, length, number=1):
                    for _ in range(number):
                        transform_fn(input, window=window, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                    power=power, n_iter=n_iter, momentum=momentum, length=length, rand_init=False)

                kwargs = {"input": input, "window": window, "n_fft": n_fft,
                          "hop_length": hop_length, "win_length": win_length, "power": power, "n_iter": n_iter,
                          "momentum": momentum, "length": length}
                usages = []
                for _ in range(repeat):
                    ret = memory_usage((run_fn, [], kwargs), **memusage_kwargs)
                    usages.append(np.max(ret) - ret[0])

                print(f"{np.mean(usages)} +- {sem(usages)}")
                results[("griffinlim", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(usages), sem(usages))

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")

        input = specgram_np.astype(dtype, copy=True)

        def run_fn(input, hop_length, n_iter, momentum, length, number=1):
            for _ in range(number):
                librosa.griffinlim(input, n_iter=n_iter, hop_length=hop_length, momentum=momentum, init=None, length=length)

        kwargs = {"input": input, "n_iter": n_iter, "hop_length": hop_length,
                  "momentum": momentum, "length": length}
        usages = []
        for _ in range(repeat):
            ret = memory_usage((run_fn, [], kwargs), **memusage_kwargs)
            usages.append(np.max(ret) - ret[0])

        print(f"{np.mean(usages)} +- {sem(usages)}")
        results[("griffinlim", "librosa", "cpu", str(dtype), int(False))] = (np.mean(usages), sem(usages))

    print(results)
    update_results(results, "./results/results.pkl")


if __name__ == "__main__":
    main()