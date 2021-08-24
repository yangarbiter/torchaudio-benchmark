import timeit

import torchaudio
import librosa
import torch
import numpy as np
from scipy.stats import sem

from utils import update_results


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

                fn_str = "transform_fn(spec_t, rate=rate, phase_advance=phase_advance_t)"

                # To avoid the first cuda run being slow
                # https://forums.developer.nvidia.com/t/execution-time-the-first-execution-time-is-always-slow/2387
                exec(fn_str)

                res = timeit.repeat(fn_str, repeat=repeat, number=number,
                                    globals={"transform_fn": transform_fn, "spec_t": spec_t,
                                             "rate": rate, "phase_advance_t": phase_advance_t})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("phase vocoder", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

    for dtype in [np.csingle, np.cdouble]:
        print(f"[librosa cpu {dtype}]")

        transform_fn = librosa.phase_vocoder
        spec_t = torch.clone(spec).detach().cpu().numpy().astype(dtype, copy=True)
        fn_str = "transform_fn(spec_t, rate=rate, hop_length=hop_length)"

        exec(fn_str)  # for fair comparison

        res = timeit.repeat(fn_str, repeat=repeat, number=number,
                            globals={"transform_fn": transform_fn, "spec_t": spec_t,
                                     "rate": rate, "hop_length": hop_length,})
        print(f"{np.mean(res)} +- {sem(res)}")
        results[("phase vocoder", "librosa", "cpu", str(dtype), int(False))] = (np.mean(res), sem(res))

    print(results)
    update_results(results, "./results/results.pkl")


if __name__ == "__main__":
    main()