import torch
import os
import sys
import scipy.io.wavfile as wavfile
import soundfile 
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# Set up local project paths for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.input_processor import InputProcessor 

from pathlib import Path

MODEL_ID = "facebook/musicgen-small" 

PROJECT_BASE_DIR = Path("K:\\Melod_AI")
if not PROJECT_BASE_DIR.exists():
    PROJECT_BASE_DIR = Path("./melodai_data")
OUTPUT_DIR = PROJECT_BASE_DIR / "Sample Outputs"

class MusicGenerator:
    """Handles prompt compilation, model loading, and music generation using Transformers."""
    def __init__(self):
        # Set computational device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MusicGen onto device: {self.device}")

        # Initialize Hugging Face processors and model components
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(self.device)
        
        # Ensure output storage structure exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory created at: {OUTPUT_DIR}")

    def _compile_prompt(self, params: dict) -> str:
        """
        Takes structured parameters and compiles them into an optimal, detailed text prompt 
        for the MusicGen model.
        """
        mood = params.get('mood', 'neutral')
        energy = params.get('energy_level', 5)
        genre = params.get('genre', 'ambient')
        tempo = params.get('tempo', 'medium')
        instruments = params.get('instruments', 'synthesizer')
        context = params.get('context', 'background music')

        compiled_prompt = (
            f"A {genre} track, {mood} and atmospheric, "
            f"intended for {context}. "
            f"Key instrumentation: {instruments}. "
            f"The tempo should be {tempo} and the intensity/energy level is {energy} out of 10. "
            f"High quality audio."
        )
        return compiled_prompt
    
    def generate_music(self, user_text: str):
        print(f"\n--- Processing Request: '{user_text}' ---")
        
        # 1. Process natural language query into structural parameters via the LLM service
        try:
            llm_processor = InputProcessor()
            params = llm_processor.process(user_text)
            
            if hasattr(params, 'dict'):
                params_dict = params.dict()
            else:
                params_dict = params
            
            print(f"LLM extracted parameters: {params_dict}")

        except Exception as e:
            print(f"FATAL ERROR: Could not initialize or process LLM input. {e}")
            return None

        # 2. Compile structural metadata into an optimized descriptive prompt string
        musicgen_prompt = self._compile_prompt(params_dict)
        print(f"-> MusicGen Prompt: {musicgen_prompt}")

        # 3. Tokenize input prompt and transfer execution tensors to target device
        inputs = self.processor(
            text=[musicgen_prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        print("Generating audio (GPU running)...")
        # 4. Generate the audio sequence using the Transformers generation engine
        generation_outputs = self.model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            guidance_scale=3.0, 
        )
        
        # 5. CRITICAL FIX: Extract raw numerical tensor values from the complex class wrapper
        if hasattr(generation_outputs, "audio_values"):
            audio_tensor = generation_outputs.audio_values
        else:
            audio_tensor = generation_outputs

        # Isolate clean naming strings for file storage
        safe_name = user_text.replace(" ", "_").replace("'", "").replace("!", "").strip()[:30]
        output_path = str(OUTPUT_DIR / f"{safe_name}.wav")
        
        # 6. FIX SECTION: Extract batch 0, channel 0 and copy array cleanly to system memory
        audio_numpy = audio_tensor[0, 0].cpu().float().numpy()

        # 7. Save file explicitly via Scipy (Safely bypasses TorchCodec and internal path checks)
        wavfile.write(output_path, 32000, audio_numpy)
        
        print(f"Audio generation COMPLETE. Saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    # Self-contained backend pipeline validation routine
    generator = MusicGenerator()
    
    test_prompts = [
        "I need energetic music for my workout.",       
        "Something calming for meditation.",       
        "Generate a 1980s synthwave track with a dreamy, nostalgic feel for a late-night drive.", 
        "I want a slow, sad piano piece, perfect for crying into a cup of coffee.",
        "Give me an intense orchestral score, like epic movie trailer music.",
        "Funky jazz music for cooking dinner, make it upbeat and cheerful.",
        "A short, quirky electronic song with lots of unusual sound effects and a fast beat.",
        "Create a calm, ambient soundscape with wind chimes and distant bells for deep relaxation.",
        "Aggressive rock music with heavy drums and electric guitar, something to get me pumped up.",
        "Light, acoustic guitar music suitable for a wedding reception background."
    ]
   
    print(f"\n--- Starting {len(test_prompts)} Music Generation Tests ---")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[TEST {i}/{len(test_prompts)}] Running prompt: **{prompt}**")
        generator.generate_music(prompt)
    
    print("\n--- All 10 tests completed. Check the audio_outputs folder. ---")