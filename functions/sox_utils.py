import subprocess

import scipy.io.wavfile
import torch


def save_wav(path, data, sample_rate, channels_first=True):
    """Save wav file without torchaudio"""
    if channels_first:
        data = data.transpose(1, 0)
    scipy.io.wavfile.write(path, sample_rate, data.numpy())


def normalize_wav(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        pass
    elif tensor.dtype == torch.int32:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 2147483647.
        tensor[tensor < 0] /= 2147483648.
    elif tensor.dtype == torch.int16:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 32767.
        tensor[tensor < 0] /= 32768.
    elif tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) - 128
        tensor[tensor > 0] /= 127.
        tensor[tensor < 0] /= 128.
    return tensor


def load_wav(path: str, normalize=True, channels_first=True) -> torch.Tensor:
    """Load wav file without torchaudio"""
    sample_rate, data = scipy.io.wavfile.read(path)
    data = torch.from_numpy(data.copy())
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if normalize:
        data = normalize_wav(data)
    if channels_first:
        data = data.transpose(1, 0)
    return data, sample_rate


def _flattern(effects):
    if not effects:
        return effects
    if isinstance(effects[0], str):
        return effects
    return [item for sublist in effects for item in sublist]


def run_sox_effect(input_file, output_file, effect, *, output_sample_rate=None, output_bitdepth=None):
    """Run sox effects"""
    effect = _flattern(effect)
    command = ['sox', '--no-dither', input_file]
    if output_bitdepth:
        command += ['--bits', str(output_bitdepth)]
    command += [output_file] + effect
    if output_sample_rate:
        command += ['rate', str(output_sample_rate)]
    print(' '.join(command))
    subprocess.run(command, check=True)