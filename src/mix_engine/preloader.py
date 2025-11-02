import librosa
import numpy as np
import soundfile as sf


class d_Song:

    def __init__(self, file_path):
        """
        Initialize a d_Song object
        Maintains file and acts as a way to transform song in particular ways.

        :param file_path: Path to the file of the song to load
        """
        self.__file = None
        self.__tempo = None
        self.__pitch = None
        self.__chroma = None
        self.__energy = None
        self.__mel = None
        self.__onset_envelope = None
        self.__intro_tempo = None
        self.__intro_chroma = None
        self.__intro_energy = None
        self.__intro_mel = None
        self.__intro_onset_envelope = None
        self.__outro_tempo = None
        self.__outro_chroma = None
        self.__outro_energy = None
        self.__outro_mel = None
        self.__outro_onset_envelope = None
        self.__file_path = file_path
        self.__y = None
        self.__sr = None
        self.duration = None

        self.__outro_sec = None
        self.__intro_sec = None

        if file_path is not None:
            self.load(file_path)

    def load(self, file_path=None):
        """
        Load a song into the object and store metadata about this object
        """

        if file_path is not None:
            self.__file_path = file_path

        if self.__file_path is None:
            raise Exception("No file path provided")

        try:
            self.__y, self.__sr = librosa.load(self.__file_path, sr=None)
        except Exception as err:
            print(f"Error occurred: {err}")
            raise Exception("Error")

    def get_sr(self):
        if self.__y is None or self.__sr is None:
            self.load()

        return self.__sr

    def get_pitch(self):
        """
        Retrieves pitch from audio data
        """
        if self.__y is None or self.__sr is None:
            self.load()

        if self.__pitch is not None:
            return self.__pitch

        self.__pitch = librosa.core.piptrack(y=self.__y, sr=self.__sr)
        return self.__pitch

    def get_tempo(self, timeBounds=None, type="default"):
        """
        Retrieves tempo from audio data
        """
        if self.__y is None or self.__sr is None:
            self.load()

        if type == "intro":
            if self.__intro_tempo is not None:
                return self.__intro_tempo
        elif type == "outro":
            if self.__outro_tempo is not None:
                return self.__outro_tempo
        else:
            if self.__tempo is not None:
                return self.__tempo

        if timeBounds is None:
            if self.__tempo is not None:
                return self.__tempo
            tempo, _ = librosa.beat.beat_track(y=self.__y, sr=self.__sr)
            self.__tempo = float(tempo)
            return self.__tempo

        # use helper
        y_seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        # Quick sanity: need at least ~0.5s for reliable detection
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=self.__sr)

        if type == "intro":
            self.__intro_tempo = float(tempo)
            return self.__intro_tempo
        elif type == "outro":
            self.__outro_tempo = float(tempo)
            return self.__outro_tempo
        else:
            self.__tempo = float(tempo)
            return self.__tempo

    def get_chroma(self, timeBounds=None, type="default"):
        """
        Retrieves HPCP vector from audio data
        """
        if self.__y is None or self.__sr is None:
            self.load()

        if type == "intro":
            if self.__intro_chroma is not None:
                return self.__intro_chroma
        elif type == "outro":
            if self.__outro_chroma is not None:
                return self.__outro_chroma
        else:
            if self.__chroma is not None:
                return self.__chroma

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            y_seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        # Need enough samples to form a few CQT frames
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        # HPCP-like chroma using constant-Q chromagram (12 pitch classes)
        chroma = librosa.feature.chroma_cqt(
            y=y_seg, sr=self.__sr
        )  # shape: (12, T_frames)

        # Mean over time → 12-D vector
        h = chroma.mean(axis=1).astype(float)

        # L2-normalize for cosine comparisons later
        norm = np.linalg.norm(h)
        if norm > 0:
            h /= norm
        else:
            h = np.zeros(12, dtype=float)

        if type == "intro":
            self.__intro_chroma = h
            return self.__intro_chroma
        elif type == "outro":
            self.__outro_chroma = h
            return self.__outro_chroma
        else:
            self.__chroma = h
            return self.__chroma

    def get_mel(self, timeBounds=None, n_bands=5, type="default"):
        """
        Retrieves mel bands from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        if type == "intro":
            if self.__intro_mel is not None:
                return self.__intro_mel
        elif type == "outro":
            if self.__outro_mel is not None:
                return self.__outro_mel
        else:
            if self.__mel is not None:
                return self.__mel

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            y_seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        # Need enough samples to form a few CQT frames
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        bands = librosa.feature.melspectrogram(
            y=y_seg, sr=self.__sr, n_mels=n_bands, power=2.0
        )
        dist = bands.mean(axis=1).astype(float)

        s = dist.sum()

        if type == "intro":
            if s == 0:
                self.__intro_mel = np.zeros(n_bands, dtype=float)
            else:
                dist /= s
                self.__intro_mel = dist

            return self.__intro_mel
        elif type == "outro":
            if s == 0:
                self.__outro_mel = np.zeros(n_bands, dtype=float)
            else:
                dist /= s
                self.__outro_mel = dist

            return self.__outro_mel
        else:
            if s == 0:
                self.__mel = np.zeros(n_bands, dtype=float)
            else:
                dist /= s
                self.__mel = dist

            return self.__mel

    def get_stft(self, timeBounds=None, hop_length=512, n_fft=2048, win_length=None):
        """
        Retrieves short-time fourier transform from audio data
        """
        y_Seg = self.__y
        if timeBounds:
            y_Seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        D = librosa.stft(
            y_Seg, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

        freqs = librosa.fft_frequencies(sr=self.__sr, n_fft=n_fft)  # (1 + n_fft/2,)

        # map frame idx -> seconds (relative to the *segment start*)
        frame_idxs = np.arange(D.shape[1])
        frame_times = librosa.frames_to_time(
            frame_idxs,
            sr=self.__sr,
            hop_length=hop_length,
            n_fft=n_fft,
        )

        return D, freqs.astype(float), frame_times.astype(float)

    def get_energy(self, timeBounds=None, type="default"):
        """
        Retrieves energy reading from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        if type == "intro":
            if self.__intro_energy is not None:
                return self.__intro_energy
        elif type == "outro":
            if self.__outro_energy is not None:
                return self.__outro_energy
        else:
            if self.__energy is not None:
                return self.__energy

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            y_seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        # Need enough samples to form a few CQT frames
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        if type == "intro":
            self.__intro_energy = float(np.sqrt(np.mean(y_seg**2)))
            return self.__intro_energy
        elif type == "outro":
            self.__outro_energy = float(np.sqrt(np.mean(y_seg**2)))
            return self.__outro_energy
        else:
            self.__energy = float(np.sqrt(np.mean(y_seg**2)))
            return self.__energy

    def get_onset_envelope(
        self, timeBounds=None, hop_length=512, percussive=True, type="default"
    ):
        """
        Retrieves onset envelope from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        if type == "intro":
            if self.__intro_onset_envelope is not None:
                return self.__intro_onset_envelope
        elif type == "outro":
            if self.__outro_onset_envelope is not None:
                return self.__outro_onset_envelope
        else:
            if self.__onset_envelope is not None:
                return self.__onset_envelope

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            y_seg = self.get_audio_seg(timeBounds[0], timeBounds[1])

        if percussive:
            y_seg = librosa.effects.percussive(y_seg)

        env = librosa.onset.onset_strength(
            y=y_seg, sr=self.__sr, hop_length=hop_length, aggregate=np.median
        )

        if type == "intro":
            self.__intro_onset_envelope = env.astype(float)
            return self.__intro_onset_envelope
        elif type == "outro":
            self.__outro_onset_envelope = env.astype(float)
            return self.__outro_onset_envelope
        else:
            self.__onset_envelope = env.astype(float)
            return self.__onset_envelope

    def set_tempo(self, val):
        """
        Changes the playback tempo of the audio file by a given ratio.

        :param val: Desired tempo as a ratio relative to the current tempo.
                    e.g. 1.10 = +10% faster, 0.90 = 10% slower
        """
        if self.__y is None or self.__sr is None:
            self.load()

        if val <= 0:
            raise ValueError("Tempo ratio must be positive.")

        self.__y = librosa.effects.time_stretch(self.__y, rate=val)

    def set_pitch(self, pitch_shift):
        """
        Changes the pitch of an audio file.

        :param pitch_shift: Number of semitones to shift (positive = higher, negative = lower)
        """
        self.__y = librosa.effects.pitch_shift(
            y=self.__y, sr=self.__sr, n_steps=pitch_shift
        )

    def export_to_location(self, output_path):
        """
        Export the current song to a specific export_path

        :param output_path: Path to export the current stored y and sr as an audio file to
        """
        sf.write(output_path, self.__y, self.__sr)

    def compute_downbeats(self, meters=(3, 4, 5, 6, 7)):
        """
        Compute downbeats using beat tracking
        """

        if self.__y is None or self.__sr is None:
            self.load()

        # Onset envelope & beats
        onset_env = librosa.onset.onset_strength(
            y=self.__y, sr=self.__sr, aggregate=np.median
        )
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=self.__sr, trim=False, units="frames"
        )

        if beat_frames is None or len(beat_frames) == 0:
            # no beats detected – fall back to empty downbeats / duration windows later
            return np.array([]), np.array([])

        beat_times = librosa.frames_to_time(beat_frames, sr=self.__sr)

        # Strength at each beat frame
        # Guard against index overrun if beat_frames extends beyond onset_env
        valid = beat_frames < len(onset_env)
        beat_frames = beat_frames[valid]
        beat_times = beat_times[valid]
        if len(beat_times) == 0:
            return np.array([]), np.array([])

        strengths = onset_env[beat_frames].astype(float)

        best_m, best_k, best_score = None, None, -np.inf
        for m in meters:
            if m <= 0:
                continue
            # try each phase offset k for this meter
            for k in range(m):
                idx = np.arange(k, len(beat_times), m, dtype=int)
                if len(idx) == 0:
                    continue
                # mean strength works well and is length-agnostic
                score = strengths[idx].mean()
                if score > best_score:
                    best_score, best_m, best_k = score, m, k

        if best_m is None:
            return np.array([]), beat_times

        down_idx = np.arange(best_k, len(beat_times), best_m, dtype=int)
        downbeats = beat_times[down_idx]
        return downbeats, beat_times

    def compute_windows(self, N=12, fallback_sec=30, use_fallback=False):
        """
        Compute mixing windows

        :param N: number of downbeats to include in the intro.outro sections
        """

        if self.__intro_sec is not None and self.__outro_sec is not None:
            return (self.__intro_sec, self.__outro_sec)

        if self.__y is None or self.__sr is None:
            self.load()

        if use_fallback:
            dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            self.__intro_sec = [0, min(fallback_sec, dur)]
            self.__outro_sec = [max(0, dur - fallback_sec), dur]
            return self.__intro_sec, self.__outro_sec

        downBeats, _ = self.compute_downbeats()

        if len(downBeats) >= N:
            self.__intro_sec = [downBeats[0], downBeats[N]]
            self.__outro_sec = [downBeats[-(N + 1)], downBeats[-1]]
        else:

            dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            self.__intro_sec = [0, min(fallback_sec, dur)]
            self.__outro_sec = [max(0, dur - fallback_sec), dur]

        return self.__intro_sec, self.__outro_sec

    def get_audio_seg(self, l_Bound: float, r_Bound: float):
        """
        Retrieve Audio Segment specified by time bounds (in seconds).
        Returns just the audio segment y_seg as a 1D np.array.
        """

        if self.__y is None or self.__sr is None:
            self.load()

        total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))

        # ensure numeric + order
        start = float(l_Bound)
        end = float(r_Bound)
        if end < start:
            start, end = end, start

        # clamp
        start = max(0.0, min(start, total_dur))
        end = max(0.0, min(end, total_dur))

        s0 = int(round(self.__sr * start))
        s1 = int(round(self.__sr * end))

        return self.__y[s0:s1]
