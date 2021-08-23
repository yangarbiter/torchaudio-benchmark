import timeit

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem


from utils import get_whitenoise

def prepare_torchaudio(sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, device, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1).to(device, dtype)
    transform_fn = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=n_mfcc, norm='ortho',
        melkwargs={'hop_length': hop_length, 'n_fft': n_fft, 'n_mels': n_mels},
    ).to(device, dtype)
    return waveform, transform_fn

def prepare_librosa(sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, dtype):
    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1)[0].numpy().astype(dtype)
    def transform_fn(waveform):
        melspec = librosa.feature.melspectrogram(
            y=waveform, sr=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop_length,
            n_mels=n_mels, htk=True, norm=None
        )
        expected = librosa.feature.mfcc(
            S=librosa.core.spectrum.power_to_db(melspec),
            n_mfcc=n_mfcc, dct_type=2, norm='ortho'
        )
        return expected
    return waveform, transform_fn

def main():
    sample_rate = 16000
    n_fft = 400
    hop_length = 200
    n_mels=64
    n_mfcc=40
    duration = 10

    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                # TODO the first cuda run is slow
                waveform, transform_fn = prepare_torchaudio(
                    sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, device, dtype)
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('transform_fn(waveform)', repeat=5, number=100,
                                    globals={"transform_fn": transform_fn, "waveform": waveform})
                print(f"{np.mean(res)} +- {sem(res)}")

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")
        waveform, transform_fn = prepare_librosa(
            sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, dtype)
        res = timeit.repeat('transform_fn(waveform)', repeat=5, number=100,
                            globals={"transform_fn": transform_fn, "waveform": waveform})
        print(f"{np.mean(res)} +- {sem(res)}")


if __name__ == "__main__":
    main()