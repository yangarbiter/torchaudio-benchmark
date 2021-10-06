import argparse

import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import WaveRNN, wavernn
from tqdm import tqdm
import torchaudio

from wavernn_inference_wrapper import WaveRNNInferenceWrapper
from eval_utils import get_dataset, eval_results
from datasets import LJList2


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


def unwrap_distributed(state_dict):
    r"""torch.distributed.DistributedDataParallel wraps the model with an additional "module.".
    This function unwraps this layer so that the weights can be loaded on models with a single GPU.
    Args:
        state_dict: Original state_dict.
    Return:
        unwrapped_state_dict: Unwrapped state_dict.
    """

    return {k.replace('module.', ''): v for k, v in state_dict.items()}


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

    wavernn_model = WaveRNN(upsample_scales=[5, 5, 11], n_classes=2**args.n_bits, hop_length=275, n_freq=80)
    wavernn_model.load_state_dict(unwrap_distributed(torch.load(args.checkpoint_path)['state_dict']))
    wavernn_model.eval().to(device)
    wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

    dset = LJList2(root="./", metadata_path="../data/ljs_audio_text_test_filelist.txt")

    preds = []
    for i, (waveform, _, _, _) in tqdm(enumerate(dset), total=len(dset)):
        mel_specgram = transforms(waveform)
        with torch.no_grad():
            preds.append(
               wavernn_inference_model(mel_specgram.to(device),
                                       mulaw=True,
                                       batched=True).cpu()
            )

        torchaudio.save(f"wavs/wavernn_nvidia_batched/{i:04d}.wav", preds[-1], sample_rate=sample_rate)
        #exit()
    eval_results(preds, dset, sample_rate)
    import ipdb; ipdb.set_trace()


def eval_audio_files():
    dset = LJList2(root="./", metadata_path="../data/ljs_audio_text_test_filelist.txt")

    preds = []
    for i in range(250):
        #pred, sample_rate = torchaudio.load(f"wavs/wavernn_nvidia_batched/{i:04d}.wav")
        pred, sample_rate = torchaudio.load(f"wavs/wavernn_nvidia_nobatched/{i:04d}.wav")
        preds.append(pred)
    eval_results(preds, dset, sample_rate, with_stoi=True, subsample_stoi=100)


if __name__ == "__main__":
    eval_audio_files()

    #parser = argparse.ArgumentParser()

    #parser.add_argument(
    #    "--checkpoint-path",
    #    type=str,
    #    help="number of data loading workers",
    #)
    #parser.add_argument(
    #    "--n-bits",
    #    type=int,
    #    default=8,
    #    help="",
    #)

    #args = parser.parse_args()
    #main(args)
