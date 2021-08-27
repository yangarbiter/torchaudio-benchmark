import random

import joblib
from torchaudio.transforms import Resample
from tqdm import tqdm
import numpy as np
from datasets import LJSPEECHList

from pesq import pesq
from pystoi import stoi

def eval_results(preds, dset, sample_rate):
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
        if len_diff == 0:
            all_stois.append(stoi(ref[0], pred[0], sample_rate, extended=False))
        else:
            for j in range(abs(len_diff)):
                if len_diff > 0:
                    stois.append(stoi(ref[0], pred[0, j: j + ref.shape[1]], sample_rate, extended=False))
                else:
                    stois.append(stoi(ref[0, j: j + pred.shape[1]], pred[0], sample_rate, extended=False))
            all_stois.append(np.max(stois))

    assert len(pesqs_nb) == 50, len(pesqs_nb)
    assert len(pesqs_wb) == 50, len(pesqs_wb)
    assert len(all_stois) == 50, len(all_stois)
    print(np.mean(pesqs_nb))
    print(np.mean(pesqs_wb))
    print(np.mean(all_stois))
    return pesqs_nb, pesqs_wb, all_stois


def get_dataset():
    dataset = joblib.load('./fatchord/data/dataset.pkl')
    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-50:]

    val_dataset = LJSPEECHList(ids=test_ids, root="./")
    return val_dataset, test_ids