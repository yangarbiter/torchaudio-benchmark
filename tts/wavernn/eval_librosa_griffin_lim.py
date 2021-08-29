import torch
from torchaudio.transforms import MelSpectrogram, InverseMelScale
from tqdm import tqdm
import librosa

from eval_utils import eval_results, get_dataset


def main():
    torch.manual_seed(0)
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
    spec_transform = MelSpectrogram(**mel_kwargs)

    vocoder = torch.nn.Sequential(
        InverseMelScale(
            n_stft=(2048 // 2 + 1),
            n_mels=80,
            sample_rate=sample_rate,
            f_min=40.,
            mel_scale="slaney",
            norm='slaney',
        ),
    )

    dset, _ = get_dataset()

    preds = []
    for i in tqdm(range(50)):
        waveform = dset[i][0]
        specgram = spec_transform(waveform)
        specgram = vocoder(specgram).cpu().numpy()[0]
        preds.append(torch.from_numpy(
            librosa.griffinlim(
                specgram,
                n_iter=32,
                hop_length=275,
                win_length=1100,
                momentum=0.99,
                init=None,
                length=waveform.shape[1]
            ).reshape(1, -1)
        ))

    eval_results(preds, dset, sample_rate)



if __name__ == "__main__":
    main()
