import timeit

import torch
import torchaudio
from utils import get_whitenoise
import numpy as np
from scipy.stats import sem

from sox_utils import save_wav, load_wav, run_sox_effect
from utils import update_results


def get_whitenoise_with_file(sample_rate, duration):
    noise = get_whitenoise(
        sample_rate=sample_rate, duration=duration, scale_factor=0.9,
    )
    path = "./whitenoise.wav"
    save_wav(path, noise, sample_rate)
    return noise, path

def run_sox(input_file, effect):
    output_file = './expected.wav'
    run_sox_effect(input_file, output_file, [str(e) for e in effect])
    return load_wav(output_file)

def run_bandpass_biquad():
    results = {}
    repeat = 5
    number = 100

    central_freq = 1000
    q = 0.707
    const_skirt_gain = True
    sample_rate = 8000
    duration = 20

    data, path = get_whitenoise_with_file(sample_rate, duration=duration)

    # TODO extremely slow for GPU

    for device in [torch.device('cpu')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False, True]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                input = torch.clone(data).detach().to(device, dtype)
                transform_fn = torchaudio.functional.bandpass_biquad
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                fn_str = "transform_fn(input, sample_rate, central_freq, q, const_skirt_gain)"

                exec(fn_str)

                res = timeit.repeat(fn_str, repeat=repeat, number=number,
                                    globals={"transform_fn": transform_fn, "input": input, "sample_rate": sample_rate,
                                            "central_freq": central_freq, "q": q, "const_skirt_gain": const_skirt_gain})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("bandpass_biquad", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

    print(results)
    update_results(results, "./results/results.pkl")

    # extremely slow due to the sox call
    #fn_str = "run_sox(path, ['bandpass', central_freq, f'{q}q'])"
    #res = timeit.repeat(fn_str, repeat=repeat, number=number,
    #                    globals={"run_sox": run_sox, "path": path, "central_freq": central_freq, "q": q})
    #print(f"{np.mean(res)} +- {sem(res)}")

def main():
    run_bandpass_biquad()
                        

if __name__ == "__main__":
    main()