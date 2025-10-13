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
        self.__pitch = None
        self.__tempo = None
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

    def get_pitch(self):
        """
        Retrieves pitch from audio data
        """
        if self.__pitch is not None:
            return self.__pitch
        if self.__y is None or self.__sr is None:
            self.load()

        self.__pitch = librosa.core.piptrack(y=self.__y, sr=self.__sr)

        return self.__pitch

    def get_tempo(self, timeBounds=None):
        """
        Retrieves tempo from audio data
        """
        if self.__y is None or self.__sr is None:
            self.load()

        if timeBounds is None:
            if self.__tempo is not None:
                return self.__tempo
            tempo, _ = librosa.beat.beat_track(y=self.__y, sr=self.__sr)
            self.__tempo = float(tempo)

        if not (hasattr(timeBounds, "__len__") and len(timeBounds) == 2):
            raise ValueError("timeBounds must be a 2-element [start_sec, end_sec]")

        start, end = float(timeBounds[0]), float(timeBounds[1])
        if end < start:
            start, end = end, start  # swap if reversed

        # Clamp to track duration
        total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
        start = max(0.0, min(start, total_dur))
        end = max(0.0, min(end, total_dur))

        # Convert to samples and slice
        s0 = int(round(start * self.__sr))
        s1 = int(round(end * self.__sr))
        y_seg = self.__y[s0:s1]

        # Quick sanity: need at least ~0.5s for reliable detection
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=self.__sr)
        return float(tempo)

    def get_chroma(self, timeBounds=None):
        """
        Retrieves HPCP vector from audio data
        """
        if self.__y is None or self.__sr is None:
            self.load()

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            if not (hasattr(timeBounds, "__len__") and len(timeBounds) == 2):
                raise ValueError("timeBounds must be a 2-element [start_sec, end_sec]")

            start, end = float(timeBounds[0]), float(timeBounds[1])
            if end < start:
                start, end = end, start

            total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            start = max(0.0, min(start, total_dur))
            end = max(0.0, min(end, total_dur))

            s0 = int(round(start * self.__sr))
            s1 = int(round(end * self.__sr))
            y_seg = self.__y[s0:s1]

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

        return h

    def get_mel(self, timeBounds=None, n_bands=5):
        """
        Retrieves mel bands from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            if not (hasattr(timeBounds, "__len__") and len(timeBounds) == 2):
                raise ValueError("timeBounds must be a 2-element [start_sec, end_sec]")

            start, end = float(timeBounds[0]), float(timeBounds[1])
            if end < start:
                start, end = end, start

            total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            start = max(0.0, min(start, total_dur))
            end = max(0.0, min(end, total_dur))

            s0 = int(round(start * self.__sr))
            s1 = int(round(end * self.__sr))
            y_seg = self.__y[s0:s1]

        # Need enough samples to form a few CQT frames
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        bands = librosa.feature.melspectrogram(
            y=y_seg, sr=self.__sr, n_mels=n_bands, power=2.0
        )
        dist = bands.mean(axis=1).astype(float)

        s = dist.sum()

        if s == 0:
            return np.zeros(n_bands, dtype=float)

        dist /= s
        return dist

    def get_energy(self, timeBounds=None):
        """
        Retrieves energy reading from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            if not (hasattr(timeBounds, "__len__") and len(timeBounds) == 2):
                raise ValueError("timeBounds must be a 2-element [start_sec, end_sec]")

            start, end = float(timeBounds[0]), float(timeBounds[1])
            if end < start:
                start, end = end, start

            total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            start = max(0.0, min(start, total_dur))
            end = max(0.0, min(end, total_dur))

            s0 = int(round(start * self.__sr))
            s1 = int(round(end * self.__sr))
            y_seg = self.__y[s0:s1]

        # Need enough samples to form a few CQT frames
        if y_seg.size < int(0.5 * self.__sr):
            raise ValueError("Selected window too short for reliable tempo estimation.")

        return float(np.sqrt(np.mean(y_seg**2)))

    def get_onset_envelope(self, timeBounds=None, hop_length=512, percussive=True):
        """
        Retrieves onset envelope from audio data
        """

        if self.__y is None or self.__sr is None:
            self.load()

        # Select samples for full track or window
        if timeBounds is None:
            y_seg = self.__y
        else:
            if not (hasattr(timeBounds, "__len__") and len(timeBounds) == 2):
                raise ValueError("timeBounds must be a 2-element [start_sec, end_sec]")

            start, end = float(timeBounds[0]), float(timeBounds[1])
            if end < start:
                start, end = end, start

            total_dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            start = max(0.0, min(start, total_dur))
            end = max(0.0, min(end, total_dur))

            s0 = int(round(start * self.__sr))
            s1 = int(round(end * self.__sr))
            y_seg = self.__y[s0:s1]

        if percussive:
            y_seg = librosa.effects.percussive(y_seg)

        env = librosa.onset.onset_strength(
            y=y_seg, sr=self.__sr, hop_length=hop_length, aggregate=np.median
        )

        return env.astype(float)

    def set_tempo(self, val):
        pass

    def change_pitch(self, pitch_shift):
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

    def compute_windows(self, N=8, fallback_sec=30):
        """
        Compute mixing windows

        :param N: number of downbeats to include in the intro.outro sections
        """

        if self.__intro_sec is not None and self.__outro_sec is not None:
            return (self.__intro_sec, self.__outro_sec)

        if self.__y is None or self.__sr is None:
            self.load()

        downBeats, _ = self.compute_downbeats()

        if len(downBeats) >= N:
            self.__intro_sec = [downBeats[0], downBeats[N]]
            self.__outro_sec = [downBeats[-(N + 1)], downBeats[-1]]
        else:

            dur = float(librosa.get_duration(y=self.__y, sr=self.__sr))
            self.__intro_sec = [0, min(fallback_sec, dur)]
            self.__outro_sec = [max(0, dur - fallback_sec), dur]

        return self.__intro_sec, self.__outro_sec
