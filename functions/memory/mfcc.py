import sys
sys.path.append("../")

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from memory_profiler import memory_usage
from utils import get_whitenoise, update_results, memusage_kwargs


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
    results = {}
    repeat = 5
    number = 100

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

                waveform, transform_fn = prepare_torchaudio(
                    sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, device, dtype)

                if jitted:
                    transform_fn = torch.jit.script(transform_fn)

                def run_fn(waveform, number=1):
                    for _ in range(number):
                        transform_fn(waveform)
                    
                kwargs = {"waveform": waveform}
                usages = []
                for _ in range(repeat):
                    ret = memory_usage((run_fn, [], kwargs), **memusage_kwargs)
                    usages.append(np.max(ret) - ret[0])

                print(f"{np.mean(usages)} +- {sem(usages)}")
                results[("mfcc", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(usages), sem(usages))

    for dtype in [np.float32, np.float64]:
        print(f"[librosa cpu {dtype}]")
        waveform, transform_fn = prepare_librosa(
            sample_rate, duration, n_fft, hop_length, n_mels, n_mfcc, dtype)

        def run_fn(waveform, number=1):
            for _ in range(number):
                transform_fn(waveform)
            
        kwargs = {"waveform": waveform}
        usages = []
        for _ in range(repeat):
            ret = memory_usage((run_fn, [], kwargs), **memusage_kwargs)
            usages.append(np.max(ret) - ret[0])

        print(f"{np.mean(usages)} +- {sem(usages)}")
        results[("mfcc", "librosa", "cpu", str(dtype), int(False))] = (np.mean(usages), sem(usages))

    print(results)
    update_results(results, "./results/results.pkl")


if __name__ == "__main__":
    main()