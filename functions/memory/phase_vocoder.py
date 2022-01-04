import sys
sys.path.append("../")

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from memory_profiler import memory_usage
from utils import update_results, memusage_kwargs


def main():
    results = {}
    repeat = 5
    number = 10

    rate = 0.5
    hop_length = 256
    num_freq = 1025
    num_frames = 400
    torch.random.manual_seed(42)

    spec = torch.randn(num_freq, num_frames, dtype=torch.complex128)
    phase_advance = torch.linspace(
        0, np.pi * hop_length, num_freq, dtype=torch.float64)[..., None]

    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        for dtype in [(torch.complex64, torch.float32), (torch.complex128, torch.float64)]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                spec_t = torch.clone(spec).detach().to(device, dtype[0])
                phase_advance_t = torch.clone(phase_advance).detach().to(device, dtype[1])

                transform_fn = torchaudio.functional.phase_vocoder
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)

                def run_fn(spec_t, rate, phase_advance_t, number=1):
                    for _ in range(number):
                        transform_fn(spec_t, rate=rate, phase_advance=phase_advance_t)

                usages = []
                for _ in range(repeat):
                    ret = memory_usage((run_fn, [],
                                       {"spec_t": spec_t,
                                        "rate": rate,
                                        "phase_advance_t": phase_advance_t,
                                        "number": number}), **memusage_kwargs)
                    usages.append(np.max(ret) - ret[0])

                print(f"{np.mean(usages)} +- {sem(usages)}")
                results[("phase vocoder", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(usages), sem(usages))

    for dtype in [np.csingle, np.cdouble]:
        print(f"[librosa cpu {dtype}]")

        transform_fn = librosa.phase_vocoder
        spec_t = torch.clone(spec).detach().cpu().numpy().astype(dtype, copy=True)

        def run_fn(spec_t, rate, hop_length, number=1):
            for _ in range(number):
                transform_fn(spec_t, rate=rate, hop_length=hop_length)

        usages = []
        for _ in range(repeat):
            ret = memory_usage((run_fn, [],
                               {"spec_t": spec_t,
                                "rate": rate,
                                "hop_length": hop_length,
                                "number": number}), **memusage_kwargs)
            usages.append(np.max(ret) - ret[0])

        print(f"{np.mean(usages)} +- {sem(usages)}")
        results[("phase vocoder", "librosa", "cpu", str(dtype), int(False))] = (np.mean(usages), sem(usages))

    print(results)
    update_results(results, "./results/results.pkl")

    print(results)


if __name__ == "__main__":
    main()
