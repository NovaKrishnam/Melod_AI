import torch
import scipy.io.wavfile
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration


MODEL_NAME = "facebook/musicgen-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION_SECONDS = 8  # Length of music to generate
MAX_NEW_TOKENS = int(DURATION_SECONDS * 50) 


PROMPTS = [
    "upbeat happy pop music",
    "sad slow piano melody",
    "energetic electronic dance music",
    "calm peaceful ambient sounds",
    "romantic acoustic guitar",
    "intense dramatic orchestral",
    "groovy funk bass",
    "mysterious dark atmospheric"
]


print("--- Loading MusicGen Model ---")
print(f"Loading model {MODEL_NAME} on {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
sampling_rate = model.config.audio_encoder.sampling_rate
print("Model loaded successfully.")


results = []
for i, prompt in enumerate(PROMPTS):
    start_time = time.time()
    
    print(f"\n--- Generating {i+1}/{len(PROMPTS)}: '{prompt}' ---")
    
    inputs = processor(
        text=[prompt], 
        padding=True, 
        return_tensors="pt"
    ).to(DEVICE)

    audio_values = model.generate(
        **inputs, 
        do_sample=True, 
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.9
    )
    
    end_time = time.time()
    
    
    OUTPUT_FILE = f"output_{i+1}_{prompt[:15].replace(' ', '_')}.wav" 
    audio_array = audio_values[0, 0].cpu().numpy()

    scipy.io.wavfile.write(
        OUTPUT_FILE, 
        rate=sampling_rate, 
        data=audio_array
    )
    
    duration = end_time - start_time
    results.append((prompt, duration, OUTPUT_FILE))
    print(f"File saved: {OUTPUT_FILE} (Time: {duration:.2f} seconds)")

