"""
3.462336988449097
3.178738441467285
0.9579072714495004
"""

from tqdm import tqdm
import torchaudio

from eval_utils import get_dataset, eval_results

def main():
    dset, test_ids = get_dataset()
    sample_rate = 22050

    idx = [dset._flist[j][0] for j in range(len(test_ids))]

    preds = []
    for i in range(50):
        ii = test_ids.index(idx[i]) + 1
        pred, _ = torchaudio.load(f"./fatchord/model_outputs/ljspeech_mol.wavernn/797k_steps_{ii}_gen_NOT_BATCHED.wav")
        preds.append(pred)

    eval_results(preds, dset, sample_rate)


if __name__ == "__main__":
    main()
