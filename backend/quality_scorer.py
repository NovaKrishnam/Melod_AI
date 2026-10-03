import librosa
import numpy as np
import os
from typing import Dict, Any, Tuple

class QualityScorer:
    """
    Automated system for evaluating the quality of generated music tracks 
    based on audio metrics, duration accuracy, and dynamic content.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """Initializes the scorer with configuration thresholds."""
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Defines default parameters and quality thresholds."""
        return {
            'min_duration_diff': 2,        # seconds tolerance for duration accuracy (±2s)
            'max_silence_ratio': 0.1,      # Max 10% of track can be silent
            'clipping_threshold': 0.95,    # Max peak amplitude before considered clipped (0.0 to 1.0)
            'min_dynamic_range': 20,       # Minimum dB difference for dynamic range
            'target_cent_min': 1500,       # Min spectral centroid (Hz)
            'target_cent_max': 4500,       # Max spectral centroid (Hz)
            'max_flatness': 0.1,           # Max acceptable spectral flatness (for noise detection)
            
            # Overall Quality Thresholds (for Auto-Retry)
            'min_overall_score': 65,       # Minimum score (65/100)
            'max_retries': 2               # Max retries
        }

    # --- Core Scoring Method ---
    def score_audio(self, audio_file_path: str, expected_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates a comprehensive quality score for the audio file."""
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            
            if y.size == 0:
                raise ValueError("Audio file is empty or corrupted.")
            
            # Run all individual checks
            scores = {
                'audio_quality': self._check_audio_quality(y),
                'duration_accuracy': self._check_duration(y, sr, expected_params.get('duration', 0)),
                'silence_detection': self._check_silence(y, sr),
                'dynamic_range': self._check_dynamics(y),
                'frequency_balance': self._check_frequency(y, sr),
            }
            
            # Calculate and include the overall score
            scores['overall_score'] = self._calculate_overall_score(scores)
            
            return scores
            
        except FileNotFoundError:
            return {'overall_score': 0, 'error': f"File not found: {audio_file_path}"}
        except Exception as e:
            return {'overall_score': 0, 'error': str(e)}

    # --- Individual Quality Checks (Scoring logic) ---

    def _check_audio_quality(self, y: np.ndarray) -> int:
        """Checks for Clipping detection and Volume normalization."""
        peak_amplitude = np.max(np.abs(y))
        clipping_threshold = self.config['clipping_threshold']
        
        # Clipping Score
        if peak_amplitude > clipping_threshold:
            penalty = 30 * (peak_amplitude - clipping_threshold) / (1 - clipping_threshold)
            clipping_score = max(0, 100 - penalty)
        else:
            clipping_score = 100
        
        # Volume Score (penalty if too quiet)
        rms = librosa.feature.rms(y=y)[0].mean()
        volume_score = 100 if rms >= 0.05 else 50
            
        return int((clipping_score * 0.7) + (volume_score * 0.3))


    def _check_duration(self, y: np.ndarray, sr: int, expected_duration: float) -> int:
        """Checks if actual duration matches requested duration ± tolerance."""
        if expected_duration == 0:
            return 100 
            
        actual_duration = librosa.get_duration(y=y, sr=sr)
        diff = abs(actual_duration - expected_duration)
        tolerance = self.config['min_duration_diff']
        
        if diff <= tolerance:
            return 100
        
        score = max(0, 100 - 20 * (diff - tolerance))
        return int(score)


    def _check_silence(self, y: np.ndarray, sr: int) -> int:
        """Checks for long silent sections."""
        intervals = librosa.effects.split(y, top_db=60) 
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        if total_duration == 0: return 0 
        
        sound_duration = sum([(end - start) for start, end in intervals]) / sr 
        silence_duration = total_duration - sound_duration
        
        silence_ratio = silence_duration / total_duration
        max_ratio = self.config['max_silence_ratio']
        
        if silence_ratio <= max_ratio:
            return 100
        
        score = max(0, 100 - 50 * (silence_ratio - max_ratio))
        return int(score)


    def _check_dynamics(self, y: np.ndarray) -> int:
        """Checks Dynamic range (variation between loud and quiet parts)."""
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        loudest = np.percentile(S_db.flatten(), 95)
        quietest = np.percentile(S_db.flatten(), 5)
        dynamic_range = loudest - quietest
        
        min_dr = self.config['min_dynamic_range']
        
        if dynamic_range >= min_dr:
            return 100
        
        score = 100 * (dynamic_range / min_dr)
        return int(score)


    def _check_frequency(self, y: np.ndarray, sr: int) -> int:
        """Checks Spectral characteristics."""
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # Centroid Check
        mean_cent = np.mean(cent)
        target_cent_min = self.config['target_cent_min']
        target_cent_max = self.config['target_cent_max']
        cent_score = 100 if target_cent_min < mean_cent < target_cent_max else 70

        # Flatness Check
        mean_flatness = np.mean(flatness)
        max_flatness = self.config['max_flatness']
        
        if mean_flatness <= max_flatness:
            flatness_score = 100
        else:
            flatness_score = max(0, 100 - 100 * (mean_flatness - max_flatness))
            
        return int((cent_score * 0.5) + (flatness_score * 0.5))


    def _calculate_overall_score(self, scores: Dict[str, int]) -> int:
        """Calculates the weighted average of all component scores."""
        weights = {
            'audio_quality': 0.3,
            'duration_accuracy': 0.2,
            'silence_detection': 0.2,
            'dynamic_range': 0.15,
            'frequency_balance': 0.15,
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return int(round(total_score))