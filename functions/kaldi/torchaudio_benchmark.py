import sys
sys.path.append("../")
from functools import partial
import timeit

from kaldiio import WriteHelper, ReadHelper
import torchaudio
import torchaudio.compliance
import librosa
import torch
import numpy as np
from scipy.stats import sem

from utils import get_whitenoise, update_results

def encode(x):
    return (x+1) * (32768+32767) / 2 - 32768


def main():
    results = {}
    repeat = 5
    number = 1

    sample_rate = 16000
    n_fft = 400
    hop_length = 200
    duration = 3000

    waveform = get_whitenoise(sample_rate=sample_rate, duration=duration, n_channels=1)
    torchaudio.save("test.wav", waveform, sample_rate=sample_rate, encoding="PCM_S",
                    bits_per_sample=16)

    for device in [torch.device('cpu'), torch.device('cuda:1')]:
        for dtype in [torch.float32, torch.float64]:
            for jitted in [False]:
                if jitted:
                    print(f"[torchaudio {device} {dtype} jitted]")
                else:
                    print(f"[torchaudio {device} {dtype}]")

                def run(tar_fn):
                    waveform, _ = torchaudio.load("./test.wav")
                    waveform = encode(waveform.to(device, dtype))
                    ret = tar_fn(waveform).float().cpu().numpy()
                    with WriteHelper('ark:file.ark') as writer:
                        writer("test", ret)

                transform_fn = partial(torchaudio.compliance.kaldi.spectrogram,
                                       dither=1.0, energy_floor=0.0)
                #run(transform_fn)
                #with ReadHelper('ark:file.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #with ReadHelper('ark:spectrogram.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #import ipdb; ipdb.set_trace()

                # To avoid the first cuda run being slow
                # https://forums.developer.nvidia.com/t/execution-time-the-first-execution-time-is-always-slow/2387
                run(transform_fn)
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('run(tar_fn)', repeat=repeat, number=number,
                                    globals={"run": run, "tar_fn": transform_fn})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("spectrogram", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

                ###################################################################################

                transform_fn = partial(torchaudio.compliance.kaldi.mfcc,
                                       dither=1.0, energy_floor=0.0, use_energy=True)
                #run(transform_fn)
                #with ReadHelper('ark:file.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #with ReadHelper('ark:mfcc.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #import ipdb; ipdb.set_trace()
                run(transform_fn)
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('run(tar_fn)', repeat=repeat, number=number,
                                    globals={"run": run, "tar_fn": transform_fn})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("mfcc", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

                ###################################################################################

                transform_fn = partial(torchaudio.compliance.kaldi.fbank,
                                       dither=1.0, energy_floor=0.0)
                #run(transform_fn)
                #with ReadHelper('ark:file.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #with ReadHelper('ark:fbank.ark') as reader:
                #    for key, numpy_array in reader:
                #        print(key, numpy_array)
                #import ipdb; ipdb.set_trace()
                run(transform_fn)
                if jitted:
                    transform_fn = torch.jit.script(transform_fn)
                res = timeit.repeat('run(tar_fn)', repeat=repeat, number=number,
                                    globals={"run": run, "tar_fn": transform_fn})
                print(f"{np.mean(res)} +- {sem(res)}")
                results[("fbank", "torchaudio", str(device), str(dtype), int(jitted))] = (np.mean(res), sem(res))

    print(results)
    update_results(results, "./results/results.pkl")


if __name__ == "__main__":
    main()
