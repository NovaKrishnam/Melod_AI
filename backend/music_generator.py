import torch
import os
import sys
import numpy as np
from pydub import AudioSegment
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import re 
import time 
import pydub 
import scipy.io.wavfile as wavfile

from pathlib import Path

# FFmpeg Configuration
pydub.AudioSegment.converter = "K:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
pydub.AudioSegment.ffprobe = "K:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"

PROJECT_BASE_DIR = Path("K:\\Melod_AI")
if not PROJECT_BASE_DIR.exists():
    PROJECT_BASE_DIR = Path("./melodai_data")
OUTPUT_DIR = PROJECT_BASE_DIR / "Sample Outputs"
SAMPLE_RATE = 32000 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MusicGenerator:
    """A robust class for generating music with dynamic multi-model support (Task 3.3)."""

    def __init__(self, default_model: str = "facebook/musicgen-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing MusicGenerator Core on device: {self.device}...")
        
        # Track loaded models to avoid redundant reloading
        self.current_model_name = default_model
        self.processor = AutoProcessor.from_pretrained(default_model)
        self.model = MusicgenForConditionalGeneration.from_pretrained(default_model).to(self.device)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _switch_model(self, model_name: str):
        """Internal helper to swap model variants in memory."""
        if model_name != self.current_model_name:
            print(f"--- Switching model to: {model_name} ---")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.current_model_name = model_name

    def _map_parameters(self, energy_level: int, duration: int) -> dict:
        """Maps user parameters to MusicGen arguments."""
        temperature = 0.6 + (energy_level / 10) * 0.4 
        cfg_coef = 4.0 - (energy_level / 10) * 2.0 
        max_tokens = int(50 * duration)
        
        return {
            "temperature": min(max(temperature, 0.6), 1.0),
            "cfg_coef": min(max(cfg_coef, 2.0), 4.0),
            "max_new_tokens": max_tokens,
        }

    def generate(self, prompt: str, duration: int = 10, energy_level: int = 5, model_name: str = "facebook/musicgen-small") -> dict[str, any]:
        """Generates music with the specified model variant."""
        
        start_time = time.time()
        
        # T3.3 Logic: Ensure the correct model variant is loaded
        try:
            self._switch_model(model_name)
        except Exception as e:
            return {"status": "error", "message": f"Failed to switch to model {model_name}: {e}"}

        generation_params = self._map_parameters(energy_level, duration)
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)

        print(f"Generating audio with {model_name} | Params: {generation_params}")

        try:
            generation_outputs = self.model.generate(
                **inputs, 
                do_sample=True,
                guidance_scale=generation_params['cfg_coef'],
                temperature=generation_params['temperature'],
                max_new_tokens=generation_params['max_new_tokens']
            )
        except Exception as e:
            print(f"ERROR during model generation: {e}")
            return {"status": "error", "message": f"Generation failed: {e}"}

        # CRITICAL FIX: Extract raw numerical audio values from transformers output object wrapper
        if hasattr(generation_outputs, "audio_values"):
            audio_tensor = generation_outputs.audio_values
        else:
            audio_tensor = generation_outputs

        audio_array = self._apply_post_processing(audio_tensor)
        
        # File naming and saving
        safe_name = re.sub(r'[^\w\-_\. ]', '_', prompt).strip()[:40]
        timestamp = int(time.time())
        mp3_path = str(OUTPUT_DIR / f"{safe_name}_{timestamp}.mp3")
        wav_temp = str(OUTPUT_DIR / f"temp_{timestamp}.wav")

        try:
            # FIX SECTION: Save temporary wav cleanly using scipy (bypasses torchaudio & torchcodec)
            wavfile.write(wav_temp, SAMPLE_RATE, audio_array)
            
            # Convert temporary wav to target MP3 output format via pydub
            AudioSegment.from_wav(wav_temp).export(mp3_path, format="mp3", bitrate="192k")
        finally:
            # Clear out the temporary raw wav file cleanly even if conversion fails
            if os.path.exists(wav_temp):
                try:
                    os.remove(wav_temp)
                except Exception as cleanup_error:
                    print(f"Failed to clean up temporary wav file: {cleanup_error}")

        end_time = time.time()
        
        return {
            "status": "success",
            "file_path": mp3_path,
            "duration_s": duration,
            "time_taken_s": round(end_time - start_time, 2),
            "model_used": model_name,
            "generation_params": generation_params
        }

    def _apply_post_processing(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Volume normalization and fade effects."""
        # Fix extraction dimension mismatch from transformers 3D tensor output
        audio_array = audio_tensor[0, 0].cpu().float().numpy()
        
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array * (0.707 / peak) 
            
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(), 
            frame_rate=SAMPLE_RATE,
            sample_width=audio_int16.dtype.itemsize,
            channels=1
        )

        fade_ms = 1000
        processed_segment = audio_segment.fade_in(fade_ms).fade_out(fade_ms)
        return np.array(processed_segment.get_array_of_samples()).astype(np.float32) / 32767.0