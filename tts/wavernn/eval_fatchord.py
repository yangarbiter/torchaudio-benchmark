import random

import torch
from tqdm import tqdm
import joblib
import torchaudio
import numpy as np

from pesq import pesq
from pystoi import stoi

from datasets import LJSPEECHList

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = joblib.load('./fatchord/data/dataset.pkl')
    dataset_ids = [x[0] for x in dataset]
    random.seed(1234)
    random.shuffle(dataset_ids)
    test_ids = dataset_ids[-50:]

    dset = LJSPEECHList(test_ids, "./")

    all_stois, pesqs_wb, pesqs_nb = [], [], []
    for i in tqdm(range(1, 51)):
        pred, sample_rate = torchaudio.load(f"./fatchord/model_outputs/ljspeech_mol.wavernn/797k_steps_{i}_gen_batched_target11000_overlap550.wav")
        idx = [dset._flist[j][0] for j in range(len(test_ids))]
        ref, sample_rate, _, _ = dset[idx.index(test_ids[i-1])]


        resampler = torchaudio.transforms.Resample(sample_rate, 16000, dtype=ref.dtype)
        re_pred = resampler(pred).numpy()
        re_ref = resampler(ref).numpy()

        pesqs_nb.append(pesq(16000, re_ref[0], re_pred[0], 'nb'))
        pesqs_wb.append(pesq(16000, re_ref[0], re_pred[0], 'wb'))


        pred, ref = pred.numpy(), ref.numpy()
        len_diff = pred.shape[1] - ref.shape[1]
        stois = []
        for j in range(abs(len_diff)):
            if len_diff > 0:
                stois.append(stoi(ref[0], pred[0, j: j + ref.shape[1]], sample_rate, extended=False))
            else:
                stois.append(stoi(ref[0, j: j + pred.shape[1]], pred[0], sample_rate, extended=False))
        all_stois.append(np.max(stois))
        print(np.max(stois))

    print(np.mean(pesqs_nb))
    print(np.mean(pesqs_wb))
    print(np.mean(all_stois))


if __name__ == "__main__":
    main()