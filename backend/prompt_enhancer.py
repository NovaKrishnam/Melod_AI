import random
import re
from typing import Dict, List


MOOD_TEMPLATES: Dict[str, str] = {
    'happy': "upbeat, cheerful, major key, bright instruments, positive vibes.",
    'sad': "melancholic, slow tempo, minor key, reflective, somber mood, minimal drums.",
    'energetic': "high energy, fast-paced, driving beat, powerful dynamics.",
    'calm': "ambient textures, slow, ethereal, wide stereo field, gentle.",
    'romantic': "warm pads, delicate piano melody, lush strings, expressive, major seventh chords.",
    'dark': "heavy bass, aggressive rhythm, minor key, distorted effects, industrial sound design."
}


class PromptEnhancer:
    def __init__(self, templates: Dict[str, str] = MOOD_TEMPLATES):
        
        self.templates = templates
        self.default_energy = 5
        self.default_tempo = 'medium'

    
    def enhance_prompt(self, params: Dict[str, any], num_variations: int = 1) -> List[str]:
        pass

    def _create_base_prompt(self, params: Dict[str, any]) -> str:
        """Takes structured parameters and creates one detailed, musically rich prompt."""
        
        # 1. Pull values with defaults
        mood = params.get('mood', 'neutral').lower()
        genre = params.get('genre', 'ambient')
        instruments = params.get('instruments', 'synthesizer and pad')
        context = params.get('context', 'background music')
        energy = params.get('energy_level', self.default_energy)
        tempo = params.get('tempo', self.default_tempo)

        
        mood_descriptor = self.templates.get(mood, f"a {mood} and dynamic atmosphere.")
        
       
        base_prompt = (
            f"A high quality **{genre}** track. "
            f"Musical Description: {mood_descriptor} "
            f"Instrumentation: **{instruments}** is key. "
            f"Dynamics and Tempo: The track features a **{tempo}** rhythm, "
            f"with an energy level of **{energy}** out of 10. "
            f"Intended for the context of **{context}**."
        )
        return base_prompt

    def enhance_prompt(self, params: Dict[str, any], num_variations: int = 3) -> List[str]:
        """
        Generates multiple enriched prompt versions for diversity in music generation.
        """
        base_prompt = self._create_base_prompt(params)
        final_prompts = [base_prompt]

        
        if num_variations > 1:
            for _ in range(num_variations - 1):
                temp_prompt = base_prompt
                
                
                if random.random() < 0.5:
                    temp_prompt = temp_prompt.replace("A high quality", "A studio quality")
                
                
                adjectives = ['lo-fi', 'punchy', 'smooth', 'complex']
                if random.random() < 0.5:
                    temp_prompt = temp_prompt.replace(f"**{params['genre']}**", f"**{random.choice(adjectives)} {params['genre']}**", 1)
                
                
                if random.random() < 0.3:
                    # Move context to the front
                    context_clause = f"Intended for the context of **{params['context']}**."
                    temp_prompt = temp_prompt.replace(context_clause, "").strip()
                    temp_prompt = context_clause + " " + temp_prompt
                
                
                final_prompts.append(temp_prompt)
        
       
        for prompt in final_prompts:
            if len(prompt) > 500: 
                print(f"Warning: Generated prompt is too long ({len(prompt)} chars).")

        return final_prompts

if __name__ == "__main__":
    
    enhancer = PromptEnhancer()

    
    llm_output_happy = {
        'mood': 'happy',
        'energy_level': 8,
        'genre': 'tropical house',
        'tempo': 'fast',
        'instruments': 'steel drums and deep bass',
        'context': 'a summer party'
    }

    llm_output_sad = {
        'mood': 'sad',
        'energy_level': 2,
        'genre': 'lo-fi beat',
        'tempo': 'slow',
        'instruments': 'muted piano and vinyl static',
        'context': 'studying on a rainy day'
    }

    print("--- Testing Prompt Enhancer: Happy Mood ---")
    happy_prompts = enhancer.enhance_prompt(llm_output_happy, num_variations=3)
    for i, p in enumerate(happy_prompts):
        print(f"Happy Prompt {i+1}: {p}")

    print("\n--- Testing Prompt Enhancer: Sad Mood ---")
    sad_prompts = enhancer.enhance_prompt(llm_output_sad, num_variations=3)
    for i, p in enumerate(sad_prompts):
        print(f"Sad Prompt {i+1}: {p}")

