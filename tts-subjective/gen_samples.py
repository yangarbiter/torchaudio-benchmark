import sys
sys.path.append("../tts/")
import torch
import torchaudio
from tqdm import tqdm
import numpy as np

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


def gen_torchaudio(ver="v1"):
	device = "cuda"
	if ver == "v1":
		from wavernn_inference_wrapper import WaveRNNInferenceWrapper
	elif ver == "v2":
		from wavernn_inference_wrapper_3 import WaveRNNInferenceWrapper

	res = torch.load("./models/torchaudio_tacotron2_wavernn_ckpt.pth")
	tacotron2 = Tacotron2(n_symbol=38).eval().to(device)
	tacotron2.load_state_dict({k.replace("module.", ""): v for k, v, in res['state_dict'].items()})

	res = torch.load("./models/parallel_wavernn_nvidia_ckpt_bs32.pt")
	wavernn_model = WaveRNN(upsample_scales=[5, 5, 11], n_classes=2**8, hop_length=275, n_freq=80)
	wavernn_model.load_state_dict({k.replace("module.", ""): v for k, v, in res['state_dict'].items()})
	wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model).eval().to(device)

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
			mel = transforms(mel)
			if ver == "v1":
				audio = wavernn_inference_model(mel, mulaw=True, batched=False).cpu()
			elif ver == "v2":
				audio = wavernn_inference_model.generate(mel.unsqueeze(0), mu_law=True, batched=False).float()

		audio = audio.reshape(1, -1)
		if ver == "v1":
			torchaudio.save(filepath=f"./audio_samples/torchaudio/torchaudio_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)
		elif ver == "v2":
			torchaudio.save(filepath=f"./audio_samples/torchaudio2/torchaudio2_{sample_no:04d}.wav", src=audio, sample_rate=sample_rate)


def main():
	#gen_torchaudio()
	gen_torchaudio("v2")


if __name__ == "__main__":
    main()
