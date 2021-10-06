import argparse

import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import WaveRNN, wavernn
from tqdm import tqdm

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

class SpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.log(torch.clamp(input, min=1e-5))

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 22050

    mel_kwargs = {
        'sample_rate': sample_rate,
        'n_fft': 1024,
        'f_min': 0.,
        'f_max': 8000.,
        'n_mels': 80,
        'win_length': 1024,
        'hop_length': 256,
        'mel_scale': 'slaney',
        'norm': 'slaney',
        'power': 1,
    }
    transforms = torch.nn.Sequential(
        MelSpectrogram(**mel_kwargs),
        SpectralNormalization(),
    )

    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    dset = LJList2(root="./", metadata_path="../data/ljs_audio_text_test_filelist.txt")


    preds = []
    for (waveform, _, _, _) in tqdm(dset):
        mel_specgram = transforms(waveform)
        with torch.no_grad():
            preds.append(
               waveglow.infer(mel_specgram.to(device)).cpu()
            )

    eval_results(preds, dset, sample_rate)


if __name__ == "__main__":
    main()
