# src/mix_engine/inference.py

from __future__ import annotations

import os
from typing import List, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa

from mix_engine.dj_bridge import get_dj_mix_curves
from mix_engine.engine import mix_Engine
from mix_engine.songOrderEngine import song_Order_Engine

from mix_engine.preloader import d_Song

order_Eng = song_Order_Engine()
mix_Eng = mix_Engine()


def _get_audio(song):
    """
    Normalize access to raw audio.

    - If song is a d_Song, use its cached waveform + sr.
    - If song is a path, load with librosa.
    """
    # Case 1: your d_Song wrapper
    if isinstance(song, d_Song):
        y = song.get_Y()
        sr = song.get_sr()
        if y is None or sr is None:
            raise ValueError("d_Song has no loaded audio or sample rate")
        return y, sr

    # Case 2: raw numpy array (we don't expect this here)
    if isinstance(song, np.ndarray):
        raise ValueError("Need sample rate for raw numpy audio (got ndarray only)")

    # Case 3: assume it's a path-like, load with librosa
    y, sr = librosa.load(song, sr=None, mono=True)
    return y, sr


def _export_discrete_segments(
    d_songs: List,
    seams: List[dict],
    out_dir: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Export *discrete* segments in play order:

      song0_intro, seam0, song1_mid_after_seam0, seam1, ...

    Returns:
      - segment_files: list of filepaths in playback order
      - segment_meta: metadata for each segment
    """
    os.makedirs(out_dir, exist_ok=True)

    segment_files: List[str] = []
    segment_meta: List[Dict] = []

    if not d_songs:
        return segment_files, segment_meta

    # If there are no seams, just export the first song as one segment.
    if not seams:
        y0, sr0 = _get_audio(d_songs[0])
        out_path = os.path.join(out_dir, "segment_00_song0_full.wav")
        sf.write(out_path, y0, sr0)
        segment_files.append(out_path)
        segment_meta.append(
            {
                "index": 0,
                "type": "song_full",
                "song_index": 0,
                "path": out_path,
            }
        )
        return segment_files, segment_meta

    sr = seams[0]["sr"]

    # ---------- 1) First song: intro (0 -> first seam's window_a[0]) ----------
    first_song = d_songs[0]
    y_first, sr_first = _get_audio(first_song)
    assert sr_first == sr, f"Sample rate mismatch: {sr_first} vs {sr}"

    first_a_start = seams[0]["window_a"][0]  # seconds
    first_intro_end = int(first_a_start * sr)

    seg_idx = 0

    intro_path = os.path.join(out_dir, f"segment_{seg_idx:02d}_song0_intro.wav")
    sf.write(intro_path, y_first[:first_intro_end], sr)
    segment_files.append(intro_path)
    segment_meta.append(
        {
            "index": seg_idx,
            "type": "song_intro",
            "song_index": 0,
            "start_sec": 0.0,
            "end_sec": first_a_start,
            "path": intro_path,
        }
    )
    seg_idx += 1

    # ---------- 2) For each seam: seam + middle of next song ----------
    for i, seam in enumerate(seams):
        # seam between song A = d_songs[i] and song B = d_songs[i+1]
        b_idx = i + 1
        b_song = d_songs[b_idx]
        y_b, sr_b = _get_audio(b_song)
        assert sr_b == sr, f"Sample rate mismatch: {sr_b} vs {sr}"

        win_b_start, win_b_end = seam["window_b"]  # seconds

        # (a) seam segment
        y_overlap = seam["y_overlap"]
        # if original songs are stereo and overlap is mono, expand it
        if y_overlap.ndim == 1 and y_b.ndim == 2:
            y_overlap = np.stack([y_overlap, y_overlap], axis=-1)

        seam_path = os.path.join(out_dir, f"segment_{seg_idx:02d}_seam_{i}.wav")
        sf.write(seam_path, y_overlap, sr)

        segment_files.append(seam_path)
        segment_meta.append(
            {
                "index": seg_idx,
                "type": "seam",
                "seam_index": i,
                "path": seam_path,
                "window_a": seam.get("window_a"),
                "window_b": seam.get("window_b"),
                "tempo_ratio": seam.get("tempo_ratio"),
                "semitone_diff": seam.get("semitone_diff"),
            }
        )
        seg_idx += 1

        # (b) middle part of song B *after* its overlap
        b_overlap_end_sample = int(win_b_end * sr)

        if i < len(seams) - 1:
            # Next seam uses this same song as A
            next_seam = seams[i + 1]
            next_a_start = next_seam["window_a"][0]  # seconds on song B
            mid_end_sample = int(next_a_start * sr)
        else:
            # Last seam → go to end of final song
            mid_end_sample = y_b.shape[0]

        if mid_end_sample > b_overlap_end_sample:
            mid_path = os.path.join(
                out_dir, f"segment_{seg_idx:02d}_song{b_idx}_mid_after_seam{i}.wav"
            )
            sf.write(
                mid_path,
                y_b[b_overlap_end_sample:mid_end_sample],
                sr,
            )
            segment_files.append(mid_path)
            segment_meta.append(
                {
                    "index": seg_idx,
                    "type": "song_middle",
                    "song_index": b_idx,
                    "from_sec": win_b_end,
                    "to_sec": mid_end_sample / sr,
                    "path": mid_path,
                }
            )
            seg_idx += 1

    return segment_files, segment_meta


def run_mixer_for_playlist(
    wav_paths: List[str],
    out_dir: str,
) -> Tuple[List[str], List[dict], List[str], List[dict]]:
    """
    Given a list of WAV paths:
      1. Order with TSP (song_OrderEngine)
      2. For each adjacent pair:
          - get_dj_mix_curves
          - mix_Engine.mix_songs
          - write seam_X.wav in out_dir
      3. Export *discrete* segments in playback order:
          song0_intro, seam0, song1_mid, seam1, ...

    Returns:
      - seam_files: list of seam .wav paths (seam_XX.wav)
      - seams_meta: seam metadata
      - segment_files: list of segment .wav paths in playback order
      - segment_meta: metadata describing each segment
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Order playlist
    song_Order, cost = order_Eng.solve_tsp(wav_paths)
    paths = song_Order["path"]
    d_songs = song_Order["d_Song"]

    seams_meta: List[dict] = []
    seam_files: List[str] = []
    seams: List[dict] = []

    # 2) For each adjacent pair, get curves + mix
    for i in range(len(paths) - 1):
        prev_path = paths[i]
        next_path = paths[i + 1]
        prev_song = d_songs[i]
        next_song = d_songs[i + 1]

        print(f"\n=== Seam {i} ===")
        print(f"Prev: {prev_path}")
        print(f"Next: {next_path}")

        dj_curves = get_dj_mix_curves(prev_path, next_path)
        print(f"Got {len(dj_curves)} curve bands")

        seam = mix_Eng.mix_songs(prev_song, next_song, dj_curves)
        seams.append(seam)

        # Write seam audio (for debugging/inspection)
        seam_filename = f"seam_{i:02d}.wav"
        seam_out_path = os.path.join(out_dir, seam_filename)
        sf.write(seam_out_path, seam["y_overlap"], seam["sr"])
        seam_files.append(seam_out_path)

        # Metadata
        seams_meta.append(
            {
                "index": i,
                "prev_path": prev_path,
                "next_path": next_path,
                "tempo_ratio": seam.get("tempo_ratio"),
                "semitone_diff": seam.get("semitone_diff"),
                "window_a": seam.get("window_a"),
                "window_b": seam.get("window_b"),
            }
        )

    # 3) Build discrete playback segments:
    segment_files, segment_meta = _export_discrete_segments(d_songs, seams, out_dir)

    return seam_files, seams_meta, segment_files, segment_meta
