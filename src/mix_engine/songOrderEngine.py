import math

import numpy as np

from .preloader import d_Song


class song_Order_Engine:

    def __init__(self):
        self.__song_list = []  # array of d_Song

    def __comp_tempo_heuristic(self, bpm_1: float, bpm_2: float) -> float:
        """
        Calculates the tempo similarity heuristic
        """

        if bpm_1 <= 0 or bpm_2 <= 0:
            return 0.0

        hi = max(bpm_1, bpm_2)
        lo = min(bpm_1, bpm_2)

        log_val = math.log2(hi / lo)  # how many doublings apart (signed, real)
        k = round(log_val)  # nearest integer multiple (… -2, -1, 0, 1, 2 …)
        rem = abs(log_val - k)  # 0 = perfect multiple; 0.5 = worst fractional

        # your suggested coefficients
        a = 1.2
        b = 0.25
        p = 2.1

        fin = -(a * rem + b * (abs(k) ** p))
        return math.exp(fin)

    def __comp_tone_alignment(
        self, chroma_1: list[float], chroma_2: list[float]
    ) -> float:
        """
        Calculates the tone similarity heuristic
        """
        a = np.asarray(chroma_1)
        b = np.asarray(chroma_2)
        denom = np.linalg.norm(a) * np.linalg.norm(b)

        if denom == 0:
            return 0.0

        return float(np.dot(a, b) / denom)

    def __comp_spectral_heuristic(self, mel_1: np.ndarray, mel_2: np.ndarray) -> float:
        """
        Calculates the spectral similarity heuristic
        """

        if mel_1.shape != mel_2.shape or mel_1.size == 0:
            return 0.0

        return float(1.0 - 0.5 * (np.abs(mel_1 - mel_2).sum()))

    def __comp_energy_continuity(
        self, energy_1: float, energy_2: float, E_MAX: float = 0.25
    ) -> float:
        """
        Calculates the energy continuity heuristic
        """
        diff = 1.0 - abs(energy_1 - energy_2) / E_MAX

        if diff < 0:
            return 0.0
        return diff

    def __comp_beat_alignment(self, env_1: np.ndarray, env_2: np.ndarray) -> float:
        """
        Calculates the beat alignment heuristic
        """

        # basic guards
        if env_1 is None or env_2 is None:
            return 0.0
        e1 = np.asarray(env_1, dtype=float)
        e2 = np.asarray(env_2, dtype=float)
        L = min(e1.size, e2.size)
        if L < 8:
            return 0.0

        # match lengths by truncation (we're already using intro/outro windows)
        e1 = e1[:L]
        e2 = e2[:L]

        # z-score (scale-invariant correlation); bail if flat
        s1 = float(np.std(e1))
        s2 = float(np.std(e2))
        if s1 < 1e-8 or s2 < 1e-8:
            return 0.0
        e1 = (e1 - float(np.mean(e1))) / (s1 + 1e-12)
        e2 = (e2 - float(np.mean(e2))) / (s2 + 1e-12)

        # full cross-correlation, normalized by length
        c = np.correlate(e1, e2, mode="full") / L
        mid = len(c) // 2

        # only allow small timing nudges: ±12.5% of window length (at least a few frames)
        half = max(L // 8, 6)
        lo = max(0, mid - half)
        hi = min(len(c), mid + half + 1)

        peak = float(np.max(c[lo:hi]))  # in [-1, 1] after z-scoring
        B = 0.5 * (peak + 1.0)  # map to [0, 1]
        return float(np.clip(B, 0.0, 1.0))

    def get_similarity_det(
        self,
        d_song_1: d_Song,
        d_song_2: d_Song,
        weights: list[float] = [0.26, 0.33, 0.23, 0.08, 0.1],
    ) -> float:
        """
        Compute a DSP-based similarity score that determines the 'mixability' between input songs

        :param song_1: candidate song to be mixed into the 'outro' portion of the seam
        :param song_2: candidate song to be mixed into the 'intro' portion of the seam
        :param weights: optional parameter specifying weightings of similarity metric components
        """

        _, song_1_outro = d_song_1.compute_windows(use_fallback=False)
        song_2_intro, _ = d_song_2.compute_windows(use_fallback=False)

        # tempo alignment heuristic
        song_1_bpm = d_song_1.get_tempo(timeBounds=song_1_outro, type="outro")
        song_2_bpm = d_song_2.get_tempo(timeBounds=song_2_intro, type="intro")
        tempo_alignment = self.__comp_tempo_heuristic(song_1_bpm, song_2_bpm)

        # Tonality alignment heuristic
        chroma_1 = d_song_1.get_chroma(timeBounds=song_1_outro, type="outro")
        chroma_2 = d_song_2.get_chroma(timeBounds=song_2_intro, type="intro")
        tone_alignment = self.__comp_tone_alignment(chroma_1, chroma_2)

        # energy continuuity heuristic
        energy_1 = d_song_1.get_energy(timeBounds=song_1_outro, type="outro")
        energy_2 = d_song_2.get_energy(timeBounds=song_2_intro, type="intro")
        energy_continuity = self.__comp_energy_continuity(energy_1, energy_2)

        # spectral fit heuristic
        mel_1 = d_song_1.get_mel(timeBounds=song_1_outro, type="outro")
        mel_2 = d_song_2.get_mel(timeBounds=song_2_intro, type="intro")
        spectral_fit = self.__comp_spectral_heuristic(mel_1, mel_2)

        # beat alignment heuristic
        env_1 = d_song_1.get_onset_envelope(timeBounds=song_1_outro, type="outro")
        env_2 = d_song_2.get_onset_envelope(timeBounds=song_2_intro, type="intro")
        beat_alignment = self.__comp_beat_alignment(env_1, env_2)

        w_t = weights[0]
        w_k = weights[1]
        w_e = weights[2]
        w_f = weights[3]
        w_b = weights[4]

        return (
            (w_t * tempo_alignment)
            + (w_k * tone_alignment)
            + (w_e * energy_continuity)
            + (w_f * spectral_fit)
            + (w_b * beat_alignment)
        )

    def get_similarity_emb(self, song_1: str, song_2: str) -> float:
        """
        Compute an embedding-based similarity score that determines the 'mixability' between input songs

        :param song_1: candidate song to be mixed into the 'outro' portion of the seam
        :param song_2: candidate song to be mixed into the 'intro' portion of the seam
        """

        # TODO: Embedding based "mixability" heuristic

        return 0.0

    def __precompute(self, song: d_Song):
        """
        Pre-compute all frequently used frequency measurements; measurements will be cached by d_Song object
        """
        intro, outro = song.compute_windows(use_fallback=False)
        song.get_tempo(timeBounds=intro, type="intro")
        song.get_tempo(timeBounds=outro, type="outro")

        song.get_chroma(timeBounds=intro, type="intro")
        song.get_chroma(timeBounds=outro, type="outro")

        song.get_energy(timeBounds=intro, type="intro")
        song.get_energy(timeBounds=outro, type="outro")

        song.get_mel(timeBounds=intro, type="intro")
        song.get_mel(timeBounds=outro, type="outro")

        song.get_onset_envelope(timeBounds=intro, type="intro")
        song.get_onset_envelope(timeBounds=outro, type="outro")

    def solve_tsp(self, wav_files: list[str]) -> tuple[dict, float]:
        """
        Solve Traveling Salesman Problem to find optimal track ordering

        :param wav_files: List of wav file paths
        :return: Tuple of (ordered file paths, total path cost)
        """
        if len(wav_files) < 2:
            return wav_files, 0.0

        object_Map = {}
        for file in wav_files:
            try:
                curr = d_Song(file)
                self.__precompute(curr)
                object_Map[file] = curr
            except:
                print(f"Wav Load Failure: {file}")

        wav_files = list(object_Map.keys())
        n = len(wav_files)

        # Build cost matrix using pairwise similarity scores
        cost_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Get similarity score (higher = more mixable)
                    similarity = self.get_similarity_det(
                        object_Map[wav_files[i]], object_Map[wav_files[j]]
                    )
                    # Convert to cost (higher similarity = lower cost)
                    cost_matrix[i][j] = 1.0 - similarity

        # Simple nearest neighbor TSP solution
        visited = [False] * n
        path = [0]  # Start with first song
        visited[0] = True
        total_cost = 0.0

        current = 0
        for _ in range(n - 1):
            min_cost = float("inf")
            next_song = -1

            # Find nearest unvisited song
            for j in range(n):
                if not visited[j] and cost_matrix[current][j] < min_cost:
                    min_cost = cost_matrix[current][j]
                    next_song = j

            if next_song != -1:
                path.append(next_song)
                visited[next_song] = True
                total_cost += min_cost
                current = next_song

        # Convert indices back to file paths
        ordered_files = {
            "path": [wav_files[i] for i in path],
            "d_Song": [object_Map[wav_files[i]] for i in path],
        }

        return ordered_files, total_cost
