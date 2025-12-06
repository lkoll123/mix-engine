import torch
import pyloudnorm   as pyln
import torch
import librosa

from djtransgan.config import settings 
from djtransgan.utils  import squeeze_dim


def normalize_loudness(audio, loudness=-12): # unit: db
    meter          = pyln.Meter(settings.SR)
    if isinstance(audio, torch.Tensor):
        audio      = squeeze_dim(audio).numpy()
        measured   = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, measured, loudness)
        normalized = torch.from_numpy(normalized).unsqueeze(0)
    else:
        measured   = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, measured, loudness)
    return normalized

def normalize_peak(audio, peak_loudness=-1):
    if isinstance(audio, torch.Tensor):
        audio      = squeeze_dim(audio).numpy()
        normalized = pyln.normalize.peak(audio, -1.0)
        normalized = torch.from_numpy(normalized).unsqueeze(0)
    else:
        normalized = pyln.normalize.peak(audio, -1.0)
    return normalized

def normalize(audio, norm_type='loudness'):
    if norm_type is None:
        return audio
    norm_dict = {
        'peak'    : normalize_peak(audio),
        'loudness': normalize_loudness(audio), 
    }
    normalized = norm_dict.get(norm_type, None)
    return audio if normalized is None else normalized


def get_stretch_ratio(src, tgt):
    return tgt / src

def time_stretch(audio, ratio, sr=None):
    """
    Time-stretch using librosa instead of the rubberband CLI.

    - audio can be a torch.Tensor with shape [1, T] or np.ndarray [T]
    - ratio > 1.0 -> faster (shorter)
    - ratio < 1.0 -> slower (longer)
    """
    if ratio == 1:
        return audio

    # Pick a sample rate
    if sr is None:
        # DJtransGAN uses settings.SR as its base sample rate
        sr = getattr(settings, "SR", 44100)

    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        y = squeeze_dim(audio).detach().cpu().numpy()  # [T]
        is_tensor = True
    else:
        y = np.asarray(audio, dtype=np.float32)
        is_tensor = False

    # librosa expects mono float
    y_stretched = librosa.effects.time_stretch(y, rate=ratio)

    if is_tensor:
        return torch.from_numpy(y_stretched).unsqueeze(0)
    else:
        return y_stretched

