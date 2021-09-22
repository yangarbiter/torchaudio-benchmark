from typing import Tuple, Callable, List, Union
from pathlib import Path
import random
import os
import csv

import torchaudio
import torch
from torch import Tensor
from torch.utils.data.dataset import random_split
from torchaudio.datasets import LJSPEECH, LIBRITTS
from torchaudio.transforms import MuLawEncoding
import joblib

from processing import bits_to_normalized_waveform, normalized_waveform_to_bits


class LJList2(torch.utils.data.Dataset):
    """Create a Dataset for LJSpeech-1.1.
    """

    def __init__(self,
                 root: Union[str, Path],
                 metadata_path: Union[str, Path],
                 url: str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                 folder_in_archive: str = "wavs") -> None:

        self._parse_filesystem(root, url, folder_in_archive, metadata_path)

    def _parse_filesystem(self, root, url, folder_in_archive, metadata_path) -> None:
        root = Path(root)

        basename = os.path.basename(url)

        basename = Path(basename.split(".tar.bz2")[0])
        folder_in_archive = basename / folder_in_archive

        self._path = root / folder_in_archive

        with open(metadata_path, "r", newline='') as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, transcript, normalized_transcript)``
        """
        line = self._flist[n]
        fileid_audio, transcript = line

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        return (
            waveform,
            sample_rate,
            transcript,
            None,
        )

    def __len__(self) -> int:
        return len(self._flist)


class LJSPEECHList(torch.utils.data.Dataset):
    """Create a Dataset for LJSpeech-1.1.
    """

    def __init__(self,
                 ids: List[str],
                 root: Union[str, Path],
                 url: str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                 folder_in_archive: str = "wavs") -> None:

        self._metadata_path = Path(root) / 'LJSpeech-1.1' / 'metadata.csv'
        self._parse_filesystem(root, url, folder_in_archive, self._metadata_path)

        self._ids = ids
        id_set = set(ids)
        self._flist = [i for i in self._flist if i[0] in id_set]
        assert len(self._flist) == len(id_set)

    def _parse_filesystem(self, root, url, folder_in_archive, metadata_path) -> None:
        root = Path(root)

        basename = os.path.basename(url)

        basename = Path(basename.split(".tar.bz2")[0])
        folder_in_archive = basename / folder_in_archive

        self._path = root / folder_in_archive

        with open(metadata_path, "r", newline='') as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, transcript, normalized_transcript)``
        """
        line = self._flist[n]
        fileid, transcript, normalized_transcript = line
        fileid_audio = self._path / (fileid + ".wav")

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        return (
            waveform,
            sample_rate,
            transcript,
            normalized_transcript,
        )

    def __len__(self) -> int:
        return len(self._flist)

class MapMemoryCache(torch.utils.data.Dataset):
    r"""Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n] is not None:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)


class Processed(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, item):
        specgram = self.transforms(item[0])
        return item[0].squeeze(0), specgram


def split_process_dataset(args, transforms):
    torch.manual_seed(0)
    if args.dataset == 'ljspeech_nvidia':
        train_dataset = LJList2(root=args.file_path, metadata_path="../data/ljs_audio_text_train_filelist.txt")
        val_dataset = LJList2(root=args.file_path, metadata_path="../data/ljs_audio_text_test_filelist.txt")

    elif args.dataset == 'ljspeech':
        data = LJSPEECH(root=args.file_path, download=False)

        val_length = int(len(data) * args.val_ratio)
        lengths = [len(data) - val_length, val_length]
        train_dataset, val_dataset = random_split(data, lengths)

    elif args.dataset == 'ljspeech_fatchord':

        dataset = joblib.load('./fatchord/data/dataset.pkl')
        dataset_ids = [x[0] for x in dataset]

        random.seed(1234)
        random.shuffle(dataset_ids)

        test_ids = dataset_ids[-50:]
        train_ids = dataset_ids[:-50]

        train_dataset = LJSPEECHList(ids=train_ids, root=args.file_path)
        val_dataset = LJSPEECHList(ids=test_ids, root=args.file_path)

    elif args.dataset == 'libritts':
        train_dataset = LIBRITTS(root=args.file_path, url='train-clean-100', download=False)
        val_dataset = LIBRITTS(root=args.file_path, url='dev-clean', download=False)

    else:
        raise ValueError(f"Expected dataset: `ljspeech` or `libritts`, but found {args.dataset}")

    train_dataset = Processed(train_dataset, transforms)
    val_dataset = Processed(val_dataset, transforms)

    train_dataset = MapMemoryCache(train_dataset)
    val_dataset = MapMemoryCache(val_dataset)

    return train_dataset, val_dataset


def collate_factory(args):
    def raw_collate(batch):

        pad = (args.kernel_size - 1) // 2

        # input waveform length
        wave_length = args.hop_length * args.seq_len_factor
        # input spectrogram length
        spec_length = args.seq_len_factor + pad * 2

        # max start postion in spectrogram
        max_offsets = [x[1].shape[-1] - (spec_length + pad * 2) for x in batch]

        # random start postion in spectrogram
        spec_offsets = [random.randint(0, offset) for offset in max_offsets]
        # random start postion in waveform
        wave_offsets = [(offset + pad) * args.hop_length for offset in spec_offsets]

        waveform_combine = [
            x[0][wave_offsets[i]: wave_offsets[i] + wave_length + 1]
            for i, x in enumerate(batch)
        ]
        specgram = [
            x[1][:, spec_offsets[i]: spec_offsets[i] + spec_length]
            for i, x in enumerate(batch)
        ]

        specgram = torch.stack(specgram)
        waveform_combine = torch.stack(waveform_combine)

        waveform = waveform_combine[:, :wave_length]
        target = waveform_combine[:, 1:]

        # waveform: [-1, 1], target: [0, 2**bits-1] if loss = 'crossentropy'
        if args.loss == "crossentropy":

            if args.mulaw:
                mulaw_encode = MuLawEncoding(2 ** args.n_bits)
                waveform = mulaw_encode(waveform)
                target = mulaw_encode(target)

                waveform = bits_to_normalized_waveform(waveform, args.n_bits)

            else:
                target = normalized_waveform_to_bits(target, args.n_bits)

        return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

    return raw_collate


def raw_collate_fn(batch, kernel_size, hop_length, seq_len_factor, loss, mulaw, n_bits):

    pad = (kernel_size - 1) // 2

    # input waveform length
    wave_length = hop_length * seq_len_factor
    # input spectrogram length
    spec_length = seq_len_factor + pad * 2

    # max start postion in spectrogram
    max_offsets = [x[1].shape[-1] - (spec_length + pad * 2) for x in batch]

    # random start postion in spectrogram
    spec_offsets = [random.randint(0, offset) for offset in max_offsets]
    # random start postion in waveform
    wave_offsets = [(offset + pad) * hop_length for offset in spec_offsets]

    waveform_combine = [
        x[0][wave_offsets[i]: wave_offsets[i] + wave_length + 1]
        for i, x in enumerate(batch)
    ]
    specgram = [
        x[1][:, spec_offsets[i]: spec_offsets[i] + spec_length]
        for i, x in enumerate(batch)
    ]

    specgram = torch.stack(specgram)
    waveform_combine = torch.stack(waveform_combine)

    waveform = waveform_combine[:, :wave_length]
    target = waveform_combine[:, 1:]

    # waveform: [-1, 1], target: [0, 2**bits-1] if loss = 'crossentropy'
    if loss == "crossentropy":

        if mulaw:
            mulaw_encode = MuLawEncoding(2 ** n_bits)
            waveform = mulaw_encode(waveform)
            target = mulaw_encode(target)

            waveform = bits_to_normalized_waveform(waveform, n_bits)

        else:
            target = normalized_waveform_to_bits(target, n_bits)

    return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)
