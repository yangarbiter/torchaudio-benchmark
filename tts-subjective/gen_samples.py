import sys
sys.path.append("../tts/")
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from mkdir_p import mkdir_p

from datasets import LJSPEECHList


from torchaudio.models import Tacotron2, WaveRNN
sys.path.append("../tts/wavernn/")
from processing import NormalizeDB
from text.text_preprocessing import (
    text_to_sequence,
)

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

# inverse of the normalization done when training Tacotron2
# needed for WaveRNN and Griffin-Lim as WaveGlow also does the same
# normalization
class InverseSpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)


def gen_torchaudio(ver="v1", vocoder="wavernn"):
    device = "cuda"
    if ver == "v1":
        from wavernn_inference_wrapper import WaveRNNInferenceWrapper
    elif ver == "v2":
        from wavernn_inference_wrapper_3 import WaveRNNInferenceWrapper

    #res = torch.load("./models/wavernn_bs96_lr1.7_ep2000_rm_adjlr_ckpt.pth")
    #res = torch.load("./models/best_ljspeech_nvidia_v2_wavernn_bs96_lr1.7_ep2000_ckpt.pth")
    res = torch.load("./models/best_ljspeech_nvidia_v2_wavernn_bs96_lr1.7_ep2000_step_scheduler.pth")
    tacotron2 = Tacotron2(n_symbol=38).eval().to(device)
    tacotron2.load_state_dict({k.replace("module.", ""): v for k, v, in res['state_dict'].items()})

    if vocoder == "wavernn":
        res = torch.load("./models/parallel_wavernn_nvidia_ckpt_bs32.pt")
        wavernn_model = WaveRNN(upsample_scales=[5, 5, 11], n_classes=2**8, hop_length=275, n_freq=80)
        wavernn_model.load_state_dict({k.replace("module.", ""): v for k, v, in res['state_dict'].items()})
        wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model).eval().to(device)
    elif vocoder == "wavernn2":
        res = torch.load("./models/parallel_wavernn_ljspeech_fatchord_ckpt_bs32_ep20k.pt")
        wavernn_model = WaveRNN(upsample_scales=[5, 5, 11], n_classes=2**8, hop_length=275, n_freq=80)
        wavernn_model.load_state_dict({k.replace("module.", ""): v for k, v, in res['state_dict'].items()})
        wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model).eval().to(device)
    elif vocoder == "fatchord":
        res = torch.load("./models/wave_step850K_weights.pyt")
        del res['module.step']
        state_dict = {k.replace("module.", ""): v for k, v, in res.items()}
        state_dict = {k.replace("upsample.resnet.melresnet_model.0.batch_norm1.running_mean", "upsample.resnet.melresnet_model.1"): v for k, v, in state_dict.items()}
        state_dict = {k.replace("I.", "fc."): v for k, v, in state_dict.items()}
        state_dict = {k.replace("upsample.resnet.conv_in.", "upsample.resnet.melresnet_model.0."): v for k, v, in state_dict.items()}
        state_dict = {k.replace("upsample.resnet.batch_norm.", "upsample.resnet.melresnet_model.1."): v for k, v, in state_dict.items()}
        for i in range(10):
            state_dict = {k.replace(f"upsample.resnet.layers.{i}.conv1", f"upsample.resnet.melresnet_model.{i+3}.resblock_model.0"): v for k, v, in state_dict.items()}
            state_dict = {k.replace(f"upsample.resnet.layers.{i}.conv2", f"upsample.resnet.melresnet_model.{i+3}.resblock_model.3"): v for k, v, in state_dict.items()}
            state_dict = {k.replace(f"upsample.resnet.layers.{i}.batch_norm1", f"upsample.resnet.melresnet_model.{i+3}.resblock_model.1"): v for k, v, in state_dict.items()}
            state_dict = {k.replace(f"upsample.resnet.layers.{i}.batch_norm2", f"upsample.resnet.melresnet_model.{i+3}.resblock_model.4"): v for k, v, in state_dict.items()}
        state_dict = {k.replace(f"upsample.resnet.conv_out", f"upsample.resnet.melresnet_model.13"): v for k, v, in state_dict.items()}
        for i in [1, 3, 5]:
            state_dict = {k.replace(f"upsample.up_layers.{i}", f"upsample.upsample_layers.{i}"): v for k, v, in state_dict.items()}
        wavernn_model = WaveRNN(upsample_scales=[5, 5, 11], n_classes=2**8, hop_length=275, n_freq=80)
        wavernn_model.load_state_dict(state_dict)
        wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model).eval().to(device)
        mkdir_p("./audio_samples/tacotron2fatchord")
    elif vocoder == "griffin-lim":
        from torchaudio.transforms import GriffinLim, InverseMelScale

        inv_norm = InverseSpectralNormalization()
        inv_mel = InverseMelScale(
            n_stft=(2048 // 2 + 1), n_mels=80, sample_rate=22050,
            f_min=40., f_max=22050/2, mel_scale="slaney", norm='slaney',
        )
        griffin_lim = GriffinLim(
            n_fft=2048, power=1, hop_length=275, win_length=1100,
        )
        vocoder_model = torch.nn.Sequential(inv_norm, inv_mel, griffin_lim).to(device)
        mkdir_p("./audio_samples/tacotron2griffinlim")
    else:
        raise ValueError("vocoder not supported")


    transforms = torch.nn.Sequential(
        InverseSpectralNormalization(),
        NormalizeDB(min_level_db=-100, normalization=True),
    )

    val_dset = LJSPEECHList(root="../tts/", metadata_path="../tts/data/ljs_audio_text_test_filelist.txt")
    index = np.random.RandomState(0).choice(np.arange(len(val_dset)), replace=False, size=100)

    for sample_no, i in tqdm(enumerate(index), total=len(index)):
        (waveform, sample_rate, text, _) = val_dset[i]
        #torchaudio.save(filepath=f"./audio_samples/original/original_{sample_no:04d}.wav", src=waveform, sample_rate=sample_rate)
        sequence = text_to_sequence(text)
        lengths = torch.LongTensor([len(sequence)])
        sequences = torch.LongTensor(sequence[:]).reshape(1, -1)
        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences.to(device), lengths.to(device))

        if vocoder == "griffin-lim":
            audio = vocoder_model(mel).cpu().float()
        else:
            with torch.no_grad():
                mel = transforms(mel)
                if vocoder in ["wavernn", "wavernn2"]:
                    batched = False
                elif vocoder == "fatchord":
                    batched = True
                else:
                    raise ValueError("vocoder not supported")
                if ver == "v1":
                    audio = wavernn_inference_model(mel, mulaw=True, batched=batched).cpu()
                elif ver == "v2":
                    audio = wavernn_inference_model.generate(mel.unsqueeze(0), mu_law=True, batched=batched).float()

        audio = audio.reshape(1, -1)
        if ver == "v1" and vocoder == "wavernn":
            torchaudio.save(filepath=f"./audio_samples/torchaudio/torchaudio_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
        if vocoder == "wavernn2":
            torchaudio.save(filepath=f"./audio_samples/torchaudio3/torchaudio3_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
        elif ver == "v2" and vocoder == "wavernn":
            torchaudio.save(filepath=f"./audio_samples/torchaudio2/torchaudio2_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
        elif ver == "v1" and vocoder == "fatchord":
            torchaudio.save(filepath=f"./audio_samples/tacotron2fatchord/tacotron2fatchord_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
        elif ver == "v1" and vocoder == "griffin-lim":
            torchaudio.save(filepath=f"./audio_samples/tacotron2griffinlim/tacotron2griffinlim_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
        else:
            raise ValueError("ver/vocoder not supported")


def main():
    #gen_torchaudio()
    gen_torchaudio("v1", "wavernn2")
    #gen_torchaudio("v2")
    #gen_torchaudio("v1", "fatchord")
    #gen_torchaudio("v1", "griffin-lim")


if __name__ == "__main__":
    main()
