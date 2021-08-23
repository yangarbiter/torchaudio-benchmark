"""
Text-to-speech pipeline using Tacotron2.
"""

from functools import partial
import argparse
import random

import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from torchaudio.models import tacotron2 as pretrained_tacotron2
import librosa
from tqdm import tqdm
from scipy.stats import sem

from datasets import (
    text_mel_collate_fn,
    split_process_dataset,
    SpectralNormalization,
    InverseSpectralNormalization,
)
from text.text_preprocessing import (
    available_symbol_set,
    available_phonemizers,
    text_to_sequence,
)


def parse_args():
    r"""
    Parse commandline arguments.
    """
    from torchaudio.models.tacotron2 import _MODEL_CONFIG_AND_URLS as tacotron2_config_and_urls

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        choices=list(tacotron2_config_and_urls.keys()),
        help='[string] The name of the checkpoint to load.'
    )
    parser.add_argument(
        "--jit",
        default=False,
        action="store_true",
        help="If used, the model and inference function is jitted."
    )

    preprocessor = parser.add_argument_group('text preprocessor setup')
    preprocessor.add_argument(
        '--text-preprocessor',
        default='english_characters',
        type=str,
        choices=available_symbol_set,
        help='select text preprocessor to use.'
    )
    preprocessor.add_argument(
        '--phonemizer',
        default="DeepPhonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"'
    )
    preprocessor.add_argument(
        '--phonemizer-checkpoint',
        default="./en_us_cmudict_forward.pt",
        type=str,
        help='the path or name of the checkpoint for the phonemizer, '
             'only used when text-preprocessor is "english_phonemes"'
    )
    preprocessor.add_argument(
        '--cmudict-root',
        default="./",
        type=str,
        help='the root directory for storing CMU dictionary files'
    )

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument(
        '--sample-rate',
        default=22050,
        type=int,
        help='Sampling rate'
    )
    audio.add_argument(
        '--n-fft',
        default=1024,
        type=int,
        help='Filter length for STFT'
    )
    audio.add_argument(
        '--n-mels',
        default=80,
        type=int,
        help=''
    )
    audio.add_argument(
        '--win-length',
        default=1024,
        type=int,
        help='Window length'
    )
    audio.add_argument(
        '--hop-length',
        default=256,
        type=int,
        help='Window length'
    )
    audio.add_argument(
        '--mel-fmin',
        default=0.0,
        type=float,
        help='Minimum mel frequency'
    )
    audio.add_argument(
        '--mel-fmax',
        default=8000.0,
        type=float,
        help='Maximum mel frequency'
    )

    return parser


def unwrap_distributed(state_dict):
    r"""torch.distributed.DistributedDataParallel wraps the model with an additional "module.".
    This function unwraps this layer so that the weights can be loaded on models with a single GPU.
    Args:
        state_dict: Original state_dict.
    Return:
        unwrapped_state_dict: Unwrapped state_dict.
    """

    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def get_datasets(args):
    text_preprocessor = partial(
        text_to_sequence,
        symbol_list=args.text_preprocessor,
        phonemizer=args.phonemizer,
        checkpoint=args.phonemizer_checkpoint,
        cmudict_root=args.cmudict_root,
    )

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            f_min=args.mel_fmin,
            f_max=args.mel_fmax,
            n_mels=args.n_mels,
            mel_scale='slaney',
            normalized=False,
            power=1,
            norm='slaney',
        ),
        SpectralNormalization()
    )
    trainset, valset = split_process_dataset(
        'ljspeech', "./", 0.1, transforms, text_preprocessor)
    return trainset, valset


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths, gate_padded = batch
    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    mel_specgram_padded = to_gpu(mel_specgram_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    mel_specgram_lengths = to_gpu(mel_specgram_lengths).long()
    x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y = (mel_specgram_padded, gate_padded)
    return x, y


def main(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tacotron2 = pretrained_tacotron2(args.checkpoint_name).to(device).eval()
    if args.jit:
        tacotron2 = torch.jit.script(tacotron2)

    _, valset = get_datasets(args)
    loader_params = {
        "batch_size": 16,
        "num_workers": 8,
        "prefetch_factor": 1024,
        'persistent_workers': True,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": partial(text_mel_collate_fn, n_frames_per_step=1),
    }
    val_loader = DataLoader(valset, **loader_params)

    # reference from https://github.com/SamuelBroughton/Mel-Cepstral-Distortion/blob/master/mel-cepstral-distortion.ipynb
    def log_spec_dB_dist(x, y):
        log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
        diff = x - y
        
        return log_spec_dB_const * np.sqrt(np.inner(diff, diff))
    
    inv_norm = InverseSpectralNormalization()

    costs = []
    for batch in tqdm(val_loader):
        (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = batch_to_gpu(batch)
        with torch.no_grad():
            pred_mel_specgram, pred_mel_lengths, _ = tacotron2.infer(text_padded, text_lengths)

        pred_mel_specgram = inv_norm(pred_mel_specgram).detach().cpu().numpy()
        pred_mel_lengths = pred_mel_lengths.detach().cpu().numpy()
        mel_specgram_padded = inv_norm(mel_specgram_padded).detach().cpu().numpy()
        mel_specgram_lengths = mel_specgram_lengths.detach().cpu().numpy()
        for i in range(len(mel_specgram_padded)):
            min_cost, _ = librosa.sequence.dtw(
                mel_specgram_padded[i, :, :mel_specgram_lengths[i]],
                pred_mel_specgram[i, :, :pred_mel_lengths[i]],
                metric=log_spec_dB_dist)
            costs.append(min_cost)
    
    print(np.mean(costs))
    print(sem(costs))



if __name__ == "__main__":
    parser = parse_args()
    args, _ = parser.parse_known_args()

    main(args)