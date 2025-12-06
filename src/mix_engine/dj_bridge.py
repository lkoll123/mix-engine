import numpy as np

# --- Compat layer for old libs (madmom) on NumPy >= 2.0 ---
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]

import os
import sys

import librosa
import torch




# mix-engine root: C:\Users\Wavefront\Documents\mix-engine
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# external DJtransGAN repo: C:\Users\Wavefront\Documents\DJtransGAN
DJTRANS_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DJtransGAN"))

# Make sure the external repo is at the *front* of sys.path so we do NOT
# accidentally pick up the vendored copy under external/DJtransGAN
if DJTRANS_ROOT not in sys.path:
    sys.path.insert(0, DJTRANS_ROOT)

# --- DJtransGAN imports (mirroring script/inference.py + process.py pieces) ---
from djtransgan.config import settings
from djtransgan.dataset import select_audio_region
from djtransgan.model import get_generator
from djtransgan.process import (correct_cue, select_cue_points, sync_bpm,
                                sync_cue)
from djtransgan.utils import (check_exist, download_pretrained, get_filename,
                              load_audio, load_pt, normalize, out_audio,
                              squeeze_dim, time_to_str)
from djtransgan.process import sync_cue

# IMPORTANT: we *do not* import preprocess or estimate_beat from djtransgan.process

# Cache the generator so we don't reload the checkpoint every time
_generator = None


def safe_sync_cue(prev_audio, next_audio, prev_cues, next_cues):
    """
    Wrap DJtransGAN's sync_cue to avoid hard crashes on degenerate cases
    (e.g. division by zero in get_stretch_ratio). If sync_cue fails, we
    just return the original next_audio/next_cues (no extra alignment).
    """
    try:
        return sync_cue(prev_audio, next_audio, prev_cues, next_cues)
    except ZeroDivisionError:
        print(
            "[dj_bridge] Warning: sync_cue failed with ZeroDivisionError; "
            "falling back to unaligned next track."
        )
        return next_audio, next_cues


def get_dj_generator(g_path=None, download_if_missing=True):
    """
    Load (once) and return the DJtransGAN generator with pretrained weights.

    This mirrors what the official inference script does, except we cache
    the generator for repeated use.
    """
    global _generator

    if _generator is not None:
        return _generator

    if g_path is None:
        # In the external repo layout, pretrained lives under ./pretrained/
        g_path = os.path.join(DJTRANS_ROOT, "pretrained", "djtransgan_minmax.pt")

    if not os.path.exists(g_path):
        if download_if_missing:
            print("[DJ Bridge] Checkpoint missing, downloading pretrained weights...")
            download_pretrained()
        if not os.path.exists(g_path):
            raise FileNotFoundError(f"DJtransGAN checkpoint not found at {g_path}")

    G = get_generator()
    G.load_state_dict(load_pt(g_path))
    G.eval()

    _generator = G
    return G


# =============================================================================
# Our own beat estimator to replace the broken estimate_beat
# =============================================================================


def _estimate_beat_librosa(audio_tensor: torch.Tensor):
    """
    Replacement for DJtransGAN's estimate_beat but using librosa.

    Input:
        audio_tensor: torch.Tensor, shape (C, N) or (1, N), as returned by load_audio.
    Returns:
        (beat_idx, bpm, beat_times, downbeat_times)
        - beat_idx: np.ndarray of frame indices (we don't really use it upstream)
        - bpm: float
        - beat_times: np.ndarray of times (seconds)
        - downbeat_times: np.ndarray of times (seconds)
    """
    # audio_tensor is (channels, samples)
    if audio_tensor is None or audio_tensor.numel() == 0:
        # fallback: empty everything, bpm 0
        return np.array([]), 0.0, np.array([]), np.array([])

    y = audio_tensor.squeeze().cpu().numpy()
    sr = settings.SR

    # Basic guard: if y is silent or too short, fallback hard
    if y.size < sr:  # < 1 second
        duration = y.size / float(sr)
        return (
            np.array([]),
            0.0,
            np.array([]),
            np.arange(0.0, duration, 2.0, dtype=float),
        )

    # Use librosa's beat tracker
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, trim=False
    )  # tempo is scalar or array

    if np.isscalar(tempo):
        bpm = float(tempo)
    elif tempo is None or len(tempo) == 0:
        bpm = 0.0
    else:
        bpm = float(np.mean(tempo))

    beat_frames = np.asarray(beat_frames, dtype=int)
    if beat_frames.size == 0 or bpm <= 0.0:
        # No beats found => fallback
        duration = y.size / float(sr)
        # Put "downbeats" every 2 seconds as a last resort
        return (
            np.array([]),
            0.0,
            np.array([]),
            np.arange(0.0, duration, 2.0, dtype=float),
        )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Approximate downbeats: every 4th beat is a bar start
    if beat_times.size >= 1:
        downbeat_times = beat_times[::4]
    else:
        downbeat_times = np.array([], dtype=float)

    return beat_frames, bpm, beat_times, downbeat_times


# =============================================================================
# Our own preprocess (copy of djtransgan.process.process.preprocess, but
# calling _estimate_beat_librosa instead of the broken estimate_beat)
# =============================================================================


def _preprocess(prev_audio, next_audio, prev_cue, next_cue):
    """
    Mirror djtransgan.process.process.preprocess, but use our librosa-based
    _estimate_beat_librosa instead of the original estimate_beat.
    """
    print("[1/5] beat tracking start ...")
    _, prev_bpm, _, prev_downbeat = _estimate_beat_librosa(prev_audio)
    _, next_bpm, _, next_downbeat = _estimate_beat_librosa(next_audio)
    print("[1/5] beat tracking complete ...")

    print("[2/5] bpm matching start ...")
    # next_audio is a torch.Tensor (C, N)
    next_audio, ratio = sync_bpm(next_audio, prev_bpm, next_bpm)
    # prev_downbeat / next_downbeat are numpy arrays of times (seconds)
    if isinstance(next_downbeat, np.ndarray):
        next_downbeat = next_downbeat / ratio
    else:
        next_downbeat = np.array(next_downbeat, dtype=float) / ratio

    next_cue = correct_cue(next_downbeat, next_cue / ratio)
    prev_cue = correct_cue(prev_downbeat, prev_cue)
    print("[2/5] bpm matching complete ...")

    print("[3/5] cue point select start ...")
    prev_cues, next_cues = select_cue_points(
        prev_cue, next_cue, prev_downbeat, next_downbeat
    )
    print("[3/5] cue point select complete ...")

    print("[4/5] cue region alignment start ...")
    next_audio, next_cues = safe_sync_cue(prev_audio, next_audio, prev_cues, next_cues)
    print("[4/5] cue region alignment complete ...")

    print("[5/5] normalize start ...")
    next_audio = normalize(next_audio)
    prev_audio = normalize(prev_audio)
    print("[5/5] normalize complete ...")

    # Now mirror the original logic to build generator inputs
    prev_audio_for_g, prev_cues_for_g, (prev_cues_ori, prev_timestamps) = (
        select_audio_region(
            prev_audio,
            prev_cues,
            settings.N_TIME,
            True,
            0,
        )
    )
    next_audio_for_g, next_cues_for_g, (next_cues_ori, next_timestamps) = (
        select_audio_region(
            next_audio,
            next_cues,
            settings.N_TIME,
            True,
            1,
        )
    )

    pair_audio = [prev_audio, next_audio]
    # cue_ori = [prev_cues_ori, next_cues_ori]  # only needed for postprocess
    timestamps = [prev_timestamps, next_timestamps]

    pair_audio_for_g = [
        prev_audio_for_g.unsqueeze(0),
        next_audio_for_g.unsqueeze(0).to(torch.float32),
    ]
    cue_for_g = prev_cues_for_g.unsqueeze(0).to(torch.float32)

    return (pair_audio, timestamps), (pair_audio_for_g, cue_for_g)


# =============================================================================
# DJtransGAN → your mix_Curves
# =============================================================================


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

    s      ~ first time curve becomes non-negligible
    s + δ  ~ last time curve is non-negligible
    """
    curve = np.asarray(curve)
    T = len(curve)
    if T <= 1:
        return 0.0, 0.0

    idx = np.where(curve > 0.01)[0]
    if len(idx) == 0:
        return 0.0, 0.0

    s_idx = int(idx[0])
    e_idx = int(idx[-1])
    s = s_idx / float(T - 1)
    delta = (e_idx - s_idx) / float(T - 1)
    return float(s), float(delta)


def convert_djgan_to_mix_curves(mix_out):
    """
    Convert DJtransGAN mix_out → your mix_Curves list[dict]:

    [
      { "f_lo_bin": int, "f_hi_bin": int,
        "a_time": {"s": float, "delta": float},
        "b_time": {"s": float, "delta": float} },
      ...
    ]

    We use:
      - prev.fader → a_time (fade-out for song A)
      - next.fader → b_time (fade-in for song B)

    and use next.band to define the frequency bands.
    """

    prev_side = mix_out["prev"]
    next_side = mix_out["next"]

    # band:  (1, n_bands, 1, F)  → (n_bands, F)
    # fader: (1, n_faders, 1, T) → (n_faders, T)
    band_prev = prev_side["band"].detach().cpu().squeeze(0).squeeze(1).numpy()
    band_next = next_side["band"].detach().cpu().squeeze(0).squeeze(1).numpy()

    fader_prev = prev_side["fader"].detach().cpu().squeeze(0).squeeze(1).numpy()
    fader_next = next_side["fader"].detach().cpu().squeeze(0).squeeze(1).numpy()

    # Use incoming track ("next") bands to define freq regions
    n_bands, F = band_next.shape
    _ = band_prev  # unused but keeps symmetry

    band_edges = [extract_band_edges(band_next[i]) for i in range(n_bands)]

    def pick_curve_for_band(fader_matrix, band_idx):
        """
        fader_matrix: (n_faders, T)

        If there is only one fader, use it for all bands.
        If multiple, map band i → fader i (clamped).
        """
        n_faders, _T = fader_matrix.shape
        if n_faders == 1:
            return fader_matrix[0]
        idx = min(band_idx, n_faders - 1)
        return fader_matrix[idx]

    mix_Curves = []
    for band_idx, (f_lo, f_hi) in enumerate(band_edges):
        fade_curve_A = pick_curve_for_band(fader_prev, band_idx)
        fade_curve_B = pick_curve_for_band(fader_next, band_idx)

        s_a, delta_a = curve_to_s_delta(fade_curve_A)
        s_b, delta_b = curve_to_s_delta(fade_curve_B)

        mix_Curves.append(
            {
                "f_lo_bin": int(f_lo),
                "f_hi_bin": int(f_hi),
                "a_time": {"s": float(s_a), "delta": float(delta_a)},
                "b_time": {"s": float(s_b), "delta": float(delta_b)},
            }
        )

    return mix_Curves


# =============================================================================
# Public entrypoint: paths → mix_Curves
# =============================================================================


def get_dj_mix_curves(prev_path, next_path, prev_cue=96, next_cue=30):
    """
    Mirror DJtransGAN's inference.py, but instead of saving audio, we:

      prev_path, next_path
        -> load_audio
        -> _preprocess (beat tracking + BPM align + windowing via our
           librosa-based beat estimator)
        -> generator.infer
        -> mix_out → mix_Curves

    Returns:
        mix_Curves: list[dict] for mix_Engine.mix_songs
    """
    G = get_dj_generator()

    # 1) Load audio (same as inference.py)
    prev_audio = load_audio(prev_path)
    next_audio = load_audio(next_path)

    # 2) Preprocess (our local implementation using librosa beats)
    (pair_audio, timestamps), (pair_audio_for_g, cue_for_g) = _preprocess(
        prev_audio,
        next_audio,
        prev_cue,
        next_cue,
    )

    # 3) Run generator inference
    with torch.no_grad():
        mix_audio, mix_out = G.infer(*pair_audio_for_g, cue_region=cue_for_g)

    # 4) Convert mix_out to your band/time param dicts
    mix_Curves = convert_djgan_to_mix_curves(mix_out)
    return mix_Curves
