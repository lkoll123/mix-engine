import math

import librosa
import numpy as np

from .preloader import d_Song


class mix_Engine:

    def __init__(self):
        return

    def mix_playlist(self, playlist: list[d_Song], curves):
        seams = []
        for i in range(len(playlist) - 1):
            entry_song = playlist[i]
            outro_song = playlist[i+1]

            #TODO: calculate current pitch and tempo
            curr_pitch = None
            curr_tempo = None

            seam = self.mix_songs(entry_song, outro_song, curves)

            #TODO: revert song to original pitch and tempo
            self.blend_back(curr_pitch, curr_tempo, seam["window_b"][1], outro_song)
            seams.append(seam)
        return seams

    def mix_songs(self, song_a: d_Song, song_b: d_Song, mix_Curves: list[dict]) -> dict:
        """
        Crossfades two songs (song_a → song_b) using per-band, time-varying masks.

        Aligns tempo/pitch of song_b to song_a, extracts outro/intro windows, applies
        STFT-based frequency-time masks from `mix_Curves`, and reconstructs the
        overlapped mix via inverse STFT.

        Returns a dict containing the mixed waveform (`y_overlap`), tempo/pitch ratios,
        sample rate, and source window bounds for both songs.
        """

        def dominant_freq(pip_out):
            pitches, mags = pip_out  # shapes: (freq_bins, frames)
            idx = np.argmax(mags)
            row = idx // mags.shape[1]
            col = idx % mags.shape[1]
            f_hz = pitches[row, col]
            if f_hz == 0:
                f_hz = 440.0  # fallback
            return float(f_hz)

        song_a_tempo = song_a.get_tempo()
        song_b_tempo = song_b.get_tempo()

        song_a_pitch = dominant_freq(song_a.get_pitch())
        song_b_pitch = dominant_freq(song_b.get_pitch())

        semitone_diff = 12 * math.log2(song_a_pitch / song_b_pitch)
        tempo_ratio = song_a_tempo / song_b_tempo

        res = {}

        song_b.set_tempo(tempo_ratio)
        song_b.set_pitch(semitone_diff)

        res["tempo_ratio"] = tempo_ratio
        res["semitone_diff"] = semitone_diff

        song_a_outro = song_a.compute_windows()[1]
        song_b_intro = song_b.compute_windows()[0]

        y_A = song_a.get_audio_seg(song_a_outro[0], song_a_outro[1])
        y_B = song_b.get_audio_seg(song_b_intro[0], song_b_intro[1])

        a_Len = len(y_A)
        b_Len = len(y_B)

        min_Len = min(a_Len, b_Len)

        y_A = y_A[:min_Len]
        y_B = y_B[:min_Len]

        sr = song_a.get_sr()
        res["sr"] = sr

        stft_A, _, _ = song_a.get_stft(song_a_outro)
        stft_B, _, _ = song_b.get_stft(song_b_intro)

        mag_A, phase_A = np.abs(stft_A), np.angle(stft_A)  # (F, T_A)
        mag_B, phase_B = np.abs(stft_B), np.angle(stft_B)  # (F, T_B)

        F_A, T_A = mag_A.shape
        F_B, T_B = mag_B.shape

        F = min(F_A, F_B)
        T_min = min(T_A, T_B)

        mag_A = mag_A[:F, :T_min]
        phase_A = phase_A[:F, :T_min]

        mag_B = mag_B[:F, :T_min]
        phase_B = phase_B[:F, :T_min]

        M1 = np.zeros((F, T_min), dtype=np.float32)
        M2 = np.zeros((F, T_min), dtype=np.float32)

        v_t = np.linspace(0, 1, T_min)

        def prelu1(v, s, delta):
            shifted = v - s
            shifted = np.maximum(shifted, 0.0)
            shifted = np.minimum(shifted, 1.0)
            scaled = shifted * delta

            return np.clip(scaled, 0.0, 1.0)

        for band in mix_Curves:
            a_s = band["a_time"]["s"]
            a_delta = band["a_time"]["delta"]

            fade_out_A = 1.0 - prelu1(v_t, a_s, a_delta)

            b_s = band["b_time"]["s"]
            b_delta = band["b_time"]["delta"]

            fade_in_B = prelu1(v_t, b_s, b_delta)

            f_lo = band["f_lo_bin"]
            f_hi = band["f_hi_bin"]

            H_band = np.zeros((F,), dtype=np.float32)
            H_band[f_lo : f_hi + 1] = 1.0

            band_mask_A = H_band[:, None] * fade_out_A[None, :]
            band_mask_B = H_band[:, None] * fade_in_B[None, :]

            M1 += band_mask_A
            M2 += band_mask_B

        M1 = np.clip(M1, 0.0, 1.0)
        M2 = np.clip(M2, 0.0, 1.0)

        mag_A = np.multiply(M1, mag_A)
        mag_B = np.multiply(M2, mag_B)

        mix_A = mag_A * np.exp(1j * phase_A)
        mix_B = mag_B * np.exp(1j * phase_B)

        mix = mix_A + mix_B

        y_recon = librosa.istft(mix, hop_length=512, win_length=2048, window="hann")

        res["y_overlap"] = y_recon
        res["len_samples"] = len(y_recon)
        res["window_a"] = song_a_outro
        res["window_b"] = song_b_intro
        return res
    

    def blend_back(self, old_tempo, old_pitch, time_stamp, song, window = 10):
        #TODO: revert song to original pitch, and original tempo

        return

