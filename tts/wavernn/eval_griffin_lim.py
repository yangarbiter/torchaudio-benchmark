"""
4.169887309074402
4.011910915374756
0.9926019197686681
"""

import random

import torch
from torchaudio.transforms import GriffinLim, Spectrogram, Resample
from tqdm import tqdm
import numpy as np
import joblib

from pesq import pesq
from pystoi import stoi

from torchaudio.datasets import LJSPEECH
from datasets import LJSPEECHList


def get_dataset():
    dataset = joblib.load('./fatchord/data/dataset.pkl')
    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-50:]

    val_dataset = LJSPEECHList(ids=test_ids, root="./")
    return val_dataset


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform, sample_rate, _, _ = LJSPEECH("./")[0]

    mel_kwargs = {
        'n_fft': 2048,
        'win_length': 1100,
        'hop_length': 275,
        'power': 1,
    }
    spec_transform = Spectrogram(**mel_kwargs)

    vocoder = GriffinLim(
        n_fft=2048,
        power=1,
        hop_length=275,
        win_length=1100,
    )

    dset = get_dataset()

    preds = []
    for i in tqdm(range(50)):
        specgram = spec_transform(dset[i][0])
        with torch.no_grad():
            preds.append(vocoder(specgram.to(device)).cpu())

    all_stois, pesqs_wb, pesqs_nb = [], [], []
    for i in tqdm(range(50)):
        pred, ref = preds[i], dset[i][0]

        resampler = Resample(sample_rate, 16000, dtype=ref.dtype)
        re_pred = resampler(pred).numpy()
        re_ref = resampler(ref).numpy()

        pesqs_nb.append(pesq(16000, re_ref[0], re_pred[0], 'nb'))
        pesqs_wb.append(pesq(16000, re_ref[0], re_pred[0], 'wb'))

        pred, ref = preds[i].numpy(), dset[i][0].numpy()
        len_diff = pred.shape[1] - ref.shape[1]
        stois = []
        for j in range(abs(len_diff)):
            if len_diff > 0:
                stois.append(stoi(ref[0], pred[0, j: j + ref.shape[1]], sample_rate, extended=False))
            else:
                stois.append(stoi(ref[0, j: j + pred.shape[1]], pred[0], sample_rate, extended=False))
        all_stois.append(np.max(stois))

    print(np.mean(pesqs_nb))
    print(np.mean(pesqs_wb))
    print(np.mean(all_stois))

    #print(np.max(stois))
    #print(pesq(sample_rate, waveform, pred, 'wb'))
    #print(pesq(sample_rate, waveform, pred, 'nb'))


if __name__ == "__main__":
    main()
