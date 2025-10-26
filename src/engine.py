import numpy as np
import librosa
import math
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TrackOrderingEngine:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.track_features = {}
        
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            features = {}
            features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Mel-frequency cepstral coefficients (timbre/sound texture)
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)  # "Brightness" of the sound
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Frequency where most energy is concentrated
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # How spread out the frequencies are
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)  # How "noisy" vs "tonal" the sound is
            features['tempo'], features['beats'] = librosa.beat.beat_track(y=y, sr=sr)  # Beats per minute
            features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)  # Musical key/pitch information
            features['harmonic'], features['percussive'] = librosa.effects.hpss(y)  # Separates melodic vs rhythmic elements
            features['tonnetz'] = librosa.feature.tonnetz(y=features['harmonic'], sr=sr)  # Harmonic relationships between notes
            features['rms'] = librosa.feature.rms(y=y)  # Overall energy level
            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return {}
    
    def __comp_tempo_heuristic(self, bpm_1: float, bpm_2: float) -> float:
        """Advanced tempo similarity using logarithmic BPM matching"""
        if bpm_1 <= 0 or bpm_2 <= 0:
            return 0.0
        
        hi = max(bpm_1, bpm_2)
        lo = min(bpm_1, bpm_2)
        
        log_val = math.log2(hi / lo)  # how many doublings apart
        k = round(log_val)  # nearest integer multiple
        rem = abs(log_val - k)  # 0 = perfect multiple; 0.5 = worst fractional
        
        # coefficients for tempo matching
        a = 1.2
        b = 0.25
        p = 2.1
        
        fin = -(a * rem + b * (abs(k) ** p))
        return math.exp(fin)
    
    def __comp_tone_alignment(self, chroma_1: np.ndarray, chroma_2: np.ndarray) -> float:
        """Calculate tone similarity using chroma vectors"""
        if chroma_1.size == 0 or chroma_2.size == 0:
            return 0.0
        
        # Average chroma vectors over time
        chroma1_avg = np.mean(chroma_1, axis=1)
        chroma2_avg = np.mean(chroma_2, axis=1)
        
        denom = np.linalg.norm(chroma1_avg) * np.linalg.norm(chroma2_avg)
        if denom == 0:
            return 0.0
        
        return float(np.dot(chroma1_avg, chroma2_avg) / denom)
    
    def __comp_spectral_heuristic(self, mel_1: np.ndarray, mel_2: np.ndarray) -> float:
        """Calculate spectral similarity using mel-spectrograms"""
        if mel_1.shape != mel_2.shape or mel_1.size == 0:
            return 0.0
        
        # Normalize the difference to [0, 1] range
        max_diff = np.abs(mel_1).sum() + np.abs(mel_2).sum()
        if max_diff == 0:
            return 1.0
        
        diff = np.abs(mel_1 - mel_2).sum() / max_diff
        return float(max(0.0, 1.0 - diff))
    
    def __comp_energy_continuity(self, energy_1: float, energy_2: float, E_MAX: float = 0.25) -> float:
        """Calculate energy continuity between tracks"""
        diff = 1.0 - abs(energy_1 - energy_2) / E_MAX
        return max(0.0, diff)
    
    def __comp_beat_alignment(self, onset_env_1: np.ndarray, onset_env_2: np.ndarray) -> float:
        """Calculate beat alignment using onset envelopes"""
        if onset_env_1 is None or onset_env_2 is None:
            return 0.0
        
        e1 = np.asarray(onset_env_1, dtype=float)
        e2 = np.asarray(onset_env_2, dtype=float)
        L = min(e1.size, e2.size)
        
        if L < 8:
            return 0.0
        
        # Match lengths by truncation
        e1 = e1[:L]
        e2 = e2[:L]
        
        # Z-score normalization
        s1 = float(np.std(e1))
        s2 = float(np.std(e2))
        if s1 < 1e-8 or s2 < 1e-8:
            return 0.0
        
        e1 = (e1 - float(np.mean(e1))) / (s1 + 1e-12)
        e2 = (e2 - float(np.mean(e2))) / (s2 + 1e-12)
        
        # Cross-correlation
        c = np.correlate(e1, e2, mode="full") / L
        mid = len(c) // 2
        
        # Allow small timing nudges: ±12.5% of window length
        half = max(L // 8, 6)
        lo = max(0, mid - half)
        hi = min(len(c), mid + half + 1)
        
        peak = float(np.max(c[lo:hi]))
        B = 0.5 * (peak + 1.0)  # map to [0, 1]
        return float(np.clip(B, 0.0, 1.0))
    
    def compute_similarity(self, features1: Dict[str, np.ndarray], features2: Dict[str, np.ndarray], 
                          weights: List[float] = [0.26, 0.33, 0.23, 0.08, 0.1]) -> float:
        """Enhanced similarity computation using weighted heuristics"""
        if not features1 or not features2:
            return 0.0
        
        # Tempo alignment
        tempo_score = 0.0
        if 'tempo' in features1 and 'tempo' in features2:
            tempo_score = self.__comp_tempo_heuristic(features1['tempo'], features2['tempo'])
        
        # Tone alignment
        tone_score = 0.0
        if 'chroma' in features1 and 'chroma' in features2:
            tone_score = self.__comp_tone_alignment(features1['chroma'], features2['chroma'])
        
        # Energy continuity
        energy_score = 0.0
        if 'rms' in features1 and 'rms' in features2:
            energy1 = np.mean(features1['rms'])
            energy2 = np.mean(features2['rms'])
            energy_score = self.__comp_energy_continuity(energy1, energy2)
        
        # Spectral fit (using MFCC as mel-spectrogram proxy)
        spectral_score = 0.0
        if 'mfcc' in features1 and 'mfcc' in features2:
            mfcc1 = np.mean(features1['mfcc'], axis=1)
            mfcc2 = np.mean(features2['mfcc'], axis=1)
            spectral_score = self.__comp_spectral_heuristic(mfcc1, mfcc2)
        
        # Beat alignment (using onset envelope approximation)
        beat_score = 0.0
        if 'rms' in features1 and 'rms' in features2:
            # Use RMS as onset envelope proxy
            onset1 = features1['rms'].flatten()
            onset2 = features2['rms'].flatten()
            beat_score = self.__comp_beat_alignment(onset1, onset2)
        
        # Weighted combination
        w_tempo, w_tone, w_energy, w_spectral, w_beat = weights
        
        return (
            w_tempo * tempo_score +
            w_tone * tone_score +
            w_energy * energy_score +
            w_spectral * spectral_score +
            w_beat * beat_score
        )
    
    def solve_tsp(self, cost_matrix: np.ndarray) -> Tuple[List[int], float]:
        n = len(cost_matrix)
        if n <= 1:
            return [0], 0.0
        
        def nearest_neighbor():
            unvisited = set(range(1, n))
            path = [0]
            current = 0
            while unvisited:
                next_city = min(unvisited, key=lambda x: cost_matrix[current][x])
                path.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            return path
        
        def two_opt_improve(path):
            best_path = path[:]
            best_cost = sum(cost_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
            improved = True
            
            while improved:
                improved = False
                for i in range(1, len(path) - 1):
                    for j in range(i + 1, len(path)):
                        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                        new_cost = sum(cost_matrix[new_path[k]][new_path[k+1]] for k in range(len(new_path)-1))
                        
                        if new_cost < best_cost:
                            best_path = new_path[:]
                            best_cost = new_cost
                            improved = True
                            path = new_path[:]
            
            return best_path, best_cost
        
        initial_path = nearest_neighbor()
        optimal_path, total_cost = two_opt_improve(initial_path)
        return optimal_path, total_cost
    
    def optimize_track_order(self, wav_files: List[str]) -> Tuple[List[str], float]:
        if not wav_files:
            return [], 0.0
        if len(wav_files) == 1:
            return [wav_files[0]], 0.0
        
        print(f"Processing {len(wav_files)} tracks...")
        
        for i, wav_file in enumerate(wav_files):
            print(f"Track {i+1}/{len(wav_files)}: {wav_file}")
            self.track_features[i] = self.extract_features(wav_file)
        
        n_tracks = len(wav_files)
        cost_matrix = np.zeros((n_tracks, n_tracks))
        
        for i in range(n_tracks):
            for j in range(n_tracks):
                if i != j:
                    mixability = self.compute_similarity(self.track_features[i], self.track_features[j])
                    cost_matrix[i][j] = 1.0 - mixability
        
        print("Solving TSP...")
        optimal_path, total_cost = self.solve_tsp(cost_matrix)
        ordered_tracks = [wav_files[i] for i in optimal_path]
        
        print(f"Total cost: {total_cost:.4f}")
        return ordered_tracks, total_cost

def optimize_track_order(wav_files: List[str]) -> Tuple[List[str], float]:
    engine = TrackOrderingEngine()
    return engine.optimize_track_order(wav_files)
