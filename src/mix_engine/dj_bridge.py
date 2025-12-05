import os
import numpy as np
import torch
import librosa

# --- Wire DJtransGAN into path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DJ_ROOT = os.path.join(PROJECT_ROOT, "external", "DJtransGAN")
if DJ_ROOT not in os.sys.path:
    os.sys.path.append(DJ_ROOT)

from djtransgan.config import settings
from djtransgan.model import get_generator

# -------- minimal load_pt so we DON'T import djtransgan.utils --------

def load_pt(path, map_location=None):
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location)

# -------- cache generator so we don't reload every time --------

_generator = None

def get_dj_generator(g_path=None):
    global _generator
    if _generator is not None:
        return _generator

    if g_path is None:
        g_path = os.path.join(DJ_ROOT, "pretrained", "djtransgan_minmax.pt")

    G = get_generator()
    if not os.path.exists(g_path):
        raise FileNotFoundError(g_path)
    G.load_state_dict(load_pt(g_path))
    G.eval()
    _generator = G
    return G

# ---------------- helper functions to map DJ output -> mix_Curves ----------------

def extract_band_edges(band_row, thresh=0.5):
    """
    band_row: 1D array over frequency bins in [0,1].
    Returns (f_lo_bin, f_hi_bin).
    """
    band_row = np.asarray(band_row)
    idx = np.where(band_row > thresh)[0]
    if len(idx) == 0:
        return 0, len(band_row) - 1
    return int(idx[0]), int(idx[-1])

def curve_to_s_delta(curve):
    """
    curve: 1D array in [0,1], length T.
    Interpret time in [0,1] via index / (T-1).
    s  ~ first time it becomes non-negligible
    s+δ ~ last time it is non-negligible
    """
    curve = np.asarray(curve)
    T = len(curve)
    if T <= 1:
        return 0.0, 0.0

    idx = np.where(curve > 0.01)[0]
    if len(idx) == 0:
        return 0.0, 0.0

    s_idx = idx[0]
    e_idx = idx[-1]
    s = s_idx / float(T - 1)
    delta = (e_idx - s_idx) / float(T - 1)
    return float(s), float(delta)

def convert_djgan_to_mix_curves(mix_out):
    """
    Take DJtransGAN's mix_out dict and convert it to your mix_Curves format:

      [
        {
          "f_lo_bin": int,
          "f_hi_bin": int,
          "a_time": {"s": float, "delta": float},
          "b_time": {"s": float, "delta": float},
        },
        ...
      ]
    """
    # Use the curves for the "next" track (how the incoming song is brought in)
    side = "next"
    side_dict = mix_out[side]

    # band: (1, n_band, 1, F) -> (n_band, F)
    band = side_dict["band"].detach().cpu().squeeze(0).squeeze(1).numpy()
    # fader: (1, n_fader, 1, T) -> (n_fader, T)
    fader = side_dict["fader"].detach().cpu().squeeze(0).squeeze(1).numpy()

    n_bands, F = band.shape
    n_faders, T = fader.shape

    # 1) get band edges per band
    band_edges = [extract_band_edges(band[i]) for i in range(n_bands)]

    # 2) choose a representative fade curve over time
    lengths = [np.sum(f > 0.01) for f in fader]
    best_idx = int(np.argmax(lengths))
    fade_curve = fader[best_idx]
    s, delta = curve_to_s_delta(fade_curve)

    mix_Curves = []
    for (f_lo, f_hi) in band_edges:
        mix_Curves.append(
            {
                "f_lo_bin": f_lo,
                "f_hi_bin": f_hi,
                "a_time": {"s": s, "delta": delta},  # fade-out A
                "b_time": {"s": s, "delta": delta},  # fade-in B
            }
        )

    return mix_Curves

# ---------------- main entrypoint you call from test_pipeline ----------------

def get_dj_mix_curves(prev_path, next_path, window_seconds=16.0):
    """
    Two WAV/MP3 paths in -> list[dict] in your test_param format.
    No audio mixing here; just curve parameters.
    """
    G = get_dj_generator()
    target_sr = settings.SR

    # load mono audio at DJtransGAN SR using librosa
    y_prev, _ = librosa.load(prev_path, sr=target_sr, mono=True)
    y_next, _ = librosa.load(next_path, sr=target_sr, mono=True)

    win_samples = int(window_seconds * target_sr)

    # last window_seconds of prev, first window_seconds of next
    if len(y_prev) > win_samples:
        seg_prev = y_prev[-win_samples:]
    else:
        seg_prev = y_prev

    if len(y_next) > win_samples:
        seg_next = y_next[:win_samples]
    else:
        seg_next = y_next

    # match lengths
    L = min(len(seg_prev), len(seg_next))
    seg_prev = seg_prev[:L]
    seg_next = seg_next[:L]

    # shape: (batch=1, channel=1, samples)
    wA = torch.from_numpy(seg_prev).float().unsqueeze(0).unsqueeze(0)
    wB = torch.from_numpy(seg_next).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        mix_audio, mix_out = G.infer(wA, wB, cue_region=None)

    return convert_djgan_to_mix_curves(mix_out)
