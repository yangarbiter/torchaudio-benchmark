from functools import partial
import timeit

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from utils import get_whitenoise, update_results


def prepare_torchaudio(sample_rate, duration, n_fft, hop_length, device, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1).to(device, dtype)
    transform_fn = torchaudio.transforms.SpectralCentroid(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
    ).to(device, dtype)
    return waveform, transform_fn

def prepare_librosa(sample_rate, duration, n_fft, hop_length, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1)[0].numpy().astype(dtype)
    transform_fn = partial(librosa.feature.spectral_centroid, sr=sample_rate,
                           n_fft=n_fft, hop_length=hop_length)
    return waveform, transform_fn

def main():
    results = {}
    repeat = 5
    number = 100

    sample_rate = 16000
    n_fft = 400
    hop_length = 200
    duration = 10

    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                waveform, transform_fn = prepare_torchaudio(sample_rate, duration, n_fft, hop_length, device, dtype)

                # To avoid the first cuda run being slow
                # https://forums.developer.nvidia.com/t/execution-time-the-first-execution-time-is-always-slow/2387
                transform_fn(waveform)

                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('transform_fn(waveform)', repeat=repeat, number=number,
                                    globals={"transform_fn": transform_fn, "waveform": waveform})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("spectral centroid", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")
        waveform, transform_fn = prepare_librosa(sample_rate, duration, n_fft, hop_length, dtype)

        transform_fn(waveform)  # for fair comparison

        res = timeit.repeat('transform_fn(waveform)', repeat=repeat, number=number,
                            globals={"transform_fn": transform_fn, "waveform": waveform})
        print(f"{np.mean(res)} +- {sem(res)}")
        results[("spectral centroid", "librosa", "cpu", str(dtype), int(False))] = (np.mean(res), sem(res))

    print(results)
    update_results(results, "./results/results.pkl")


if __name__ == "__main__":
    main()