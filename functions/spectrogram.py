from functools import partial
import timeit

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem


from utils import get_whitenoise


def prepare_torchaudio(sample_rate, duration, n_fft, hop_length, device, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1).to(device, dtype)
    transform_fn = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0,
    ).to(device, dtype)
    return waveform, transform_fn

def prepare_librosa(sample_rate, duration, n_fft, hop_length, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1)[0].numpy().astype(dtype)
    transform_fn = partial(librosa.core.spectrum._spectrogram, n_fft=n_fft, hop_length=hop_length, power=2)
    return waveform, transform_fn

def main():
    sample_rate = 16000
    n_fft = 400
    hop_length = 200
    duration = 20

    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                # TODO the first cuda run is slow
                waveform, transform_fn = prepare_torchaudio(sample_rate, duration, n_fft, hop_length, device, dtype)
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('transform_fn(waveform)', repeat=5, number=100,
                                    globals={"transform_fn": transform_fn, "waveform": waveform})
                print(f"{np.mean(res)} +- {sem(res)}")

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")
        waveform, transform_fn = prepare_librosa(sample_rate, duration, n_fft, hop_length, dtype)
        res = timeit.repeat('transform_fn(waveform)', repeat=5, number=100,
                            globals={"transform_fn": transform_fn, "waveform": waveform})
        print(f"{np.mean(res)} +- {sem(res)}")


if __name__ == "__main__":
    main()