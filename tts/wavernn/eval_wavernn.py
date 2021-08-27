import argparse

import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import WaveRNN, wavernn
from tqdm import tqdm
import numpy as np

from torchaudio.datasets import LJSPEECH
from wavernn_inference_wrapper import WaveRNNInferenceWrapper
from eval_utils import get_dataset, eval_results



class NormalizeDB(torch.nn.Module):
    r"""Normalize the spectrogram with a minimum db value
    """

    def __init__(self, min_level_db, normalization):
        super().__init__()
        self.min_level_db = min_level_db
        self.normalization = normalization

    def forward(self, specgram):
        specgram = torch.log10(torch.clamp(specgram.squeeze(0), min=1e-5))
        if self.normalization:
            return torch.clamp(
                (self.min_level_db - 20 * specgram) / self.min_level_db, min=0, max=1
            )
        return specgram


def main(args):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 22050

    mel_kwargs = {
        'sample_rate': sample_rate,
        'n_fft': 2048,
        'f_min': 40.,
        'n_mels': 80,
        'win_length': 1100,
        'hop_length': 275,
        'mel_scale': 'slaney',
        'norm': 'slaney',
        'power': 1,
    }
    transforms = torch.nn.Sequential(
        MelSpectrogram(**mel_kwargs),
        NormalizeDB(min_level_db=-100, normalization=True),
    )
    mel_specgram = transforms(waveform)

    wavernn_model = WaveRNN()
    wavernn_model.load_state_dict(torch.load(args.model_path)['state_dict'])
    wavernn_model.eval().to(device)
    wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

    dset = get_dataset()
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=16,
        shuffle=False,
    )

    preds = []
    for (waveform, _, _, _) in tqdm(loader):
        mel_specgram = transforms(waveform)
        with torch.no_grad():
            preds.append(
               wavernn_inference_model(mel_specgram.to(device),
                                       mulaw=True,
                                       batched=False).cpu()
            )
            #preds.append(
            #    wavernn_inference_model.infer_batch(mel_specgram.to(device),
            #                                        mulaw=True,).cpu().numpy()
            #)
    preds = torch.cat(preds, axis=0)
    
    eval_results(preds, dset, sample_rate)

    #with torch.no_grad():
    #    pred = wavernn_inference_model(mel_specgram.to(device),
    #                                   mulaw=True,
    #                                   batched=False,
    #                                   timesteps=100,
    #                                   overlap=5,).cpu().numpy()

    #import ipdb; ipdb.set_trace()
    #ref = waveform.numpy()
    #len_diff = pred.shape[1] - ref.shape[1]
    #stois = []
    #for j in range(0, abs(len_diff), abs(len_diff) // 100):
    #    if len_diff > 0:
    #        stois.append(stoi(ref[0], pred[0, j: j + ref.shape[1]], sample_rate, extended=False))
    #    else:
    #        stois.append(stoi(ref[0, j: j + pred.shape[1]], pred[0], sample_rate, extended=False))
    #print(np.max(stois))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="number of data loading workers",
    )

    args = parser.parse_args()
    main(args)