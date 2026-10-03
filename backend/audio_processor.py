import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
from pathlib import Path

class AudioProcessor:
    """
    Task 3.7 & 3.8: Audio Post-Processing & Final Integration Suite.
    Handles advanced enhancements, room presets, format conversion, and analytics.
    """
    def __init__(self):
        pass

    def apply_effects(self, file_path: str, effect_type: str = "mastering") -> str:
        """
        Applies professional audio effects library.
        Includes Mastering, Reverb, Delay, and Stereo Widening.
        """
        audio = AudioSegment.from_file(file_path)
        
        if effect_type == "mastering":
            # EQ & Dynamics: Normalize and compress for consistent loudness
            audio = effects.normalize(audio)
            audio = effects.compress_dynamic_range(audio)
            audio = audio - 0.1 # Limiter to prevent clipping
            
        elif effect_type == "reverb":
            # Studio Reverb: Simulate physical space
            delayed = audio - 7 
            audio = audio.overlay(delayed, position=50) 

        elif effect_type == "delay":
            # Echo/Delay: Multi-tap echo for depth
            echo = audio - 12
            audio = audio.overlay(echo, position=400).overlay(echo - 5, position=800)

        elif effect_type == "stereo_widening":
            # Stereo Widening: Create an immersive soundstage
            left = audio.pan(-0.2)
            right = audio.pan(0.2)
            audio = left.overlay(right)

        output_path = file_path.replace(".mp3", f"_{effect_type}.mp3")
        audio.export(output_path, format="mp3")
        return output_path

    def apply_preset_effect(self, file_path: str, preset: str) -> str:
        """
        Task 3.8: Preset effects for quick environment modeling.
        Options: 'Studio', 'Concert Hall', 'Bedroom'.
        """
        audio = AudioSegment.from_file(file_path)
        
        if preset == "Studio":
            # Clean and professional
            audio = effects.normalize(audio)
            audio = effects.compress_dynamic_range(audio)
        elif preset == "Concert Hall":
            # High-reverb environment
            delayed = audio - 5
            audio = audio.overlay(delayed, position=100).overlay(delayed - 3, position=250)
        elif preset == "Bedroom":
            # Muffled, lo-fi intimate feel
            audio = audio.low_pass_filter(3000)
            
        output_path = file_path.replace(".mp3", f"_{preset.lower().replace(' ', '_')}.mp3")
        audio.export(output_path, format="mp3")
        return output_path

    def convert_format(self, input_path: str, target_format: str = "wav", bitrate: str = "320k") -> str:
        """
        Task 3.7: Multi-format export with quality settings and metadata tags.
        """
        audio = AudioSegment.from_file(input_path)
        output_path = str(Path(input_path).with_suffix(f".{target_format}"))
        
        # Metadata embedding for Task 3.8 handover preparation
        tags = {
            "title": "MelodAI Gen",
            "artist": "MelodAI Engine",
            "genre": "AI Composition",
            "comment": "Task 3.8 Integrated Export"
        }
        
        audio.export(output_path, format=target_format, bitrate=bitrate, tags=tags)
        return output_path

    def analyze_audio(self, file_path: str) -> dict:
        """
        Technical Analysis tools for rhythmic and harmonic transparency.
        """
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        return {
            "bpm": round(float(tempo[0]), 1),
            "detected_key": keys[key_idx],
            "sample_rate": sr
        }

    def create_spectrogram(self, file_path: str):
        """
        Task 3.7: Mel-frequency Spectrogram visualization.
        """
        y, sr = librosa.load(file_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_title('Mel-frequency Spectrogram', color='white')
        
        plt.close(fig)
        return fig