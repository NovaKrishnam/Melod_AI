import os
from openai import OpenAI
from dotenv import load_dotenv
import json 


load_dotenv()


SYSTEM_PROMPT = """
You are an intelligent music parameter extraction system. Your task is to analyze a user's music request 
and extract the following parameters into a strict JSON object.

1. 'mood' (string): Primary emotion (e.g., happy, sad, energetic, calm, peaceful).
2. 'energy_level' (integer): A number from 1 (very low) to 10 (very high) reflecting intensity.
3. 'genre' (string): The musical style (e.g., classical, rock, synthwave, ambient).
4. 'tempo' (string): Preference for speed (slow, medium, fast).
5. 'instruments' (string): Key instruments to include (e.g., piano and drums, just acoustic guitar, full orchestra).
6. 'context' (string): The situation or activity the music is for (e.g., working out, studying, meditation).

**Strictly adhere to the JSON format below. Only output the JSON object.**
{
 "mood": "...",
 "energy_level": 1,
 "genre": "...",
 "tempo": "...",
 "instruments": "...",
 "context": "..."
}
"""

class InputProcessor:
    def __init__(self, api_key=None):
        
        key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        
        if not key:
            raise ValueError("OPENAI_API_KEY not found. Please check your .env file or pass the key.")
            
        
        self.client = OpenAI(api_key=key)

    
    def process(self, user_text: str) -> dict:
        """
        Processes a user text request using the LLM to extract structured parameters.
        Returns the extracted parameters as a dictionary.
        """
        try:
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User Request: '{user_text}'"}
            ]

            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0.0 
            )
            
            
            json_str = response.choices[0].message.content.strip()
            
            
            extracted_params = json.loads(json_str)
            
            
            validated_params = self._validate_and_default(extracted_params)
            
            return validated_params

        except Exception as e:
            print(f"Error during LLM processing: {e}")
            
            return self._fallback_parsing(user_text)
    
    def _validate_and_default(self, params: dict) -> dict:
        """Ensures extracted parameters are valid and applies defaults if necessary."""
        
        
        energy = params.get('energy_level', 5) 
        try:
            energy = int(energy)
            if not 1 <= energy <= 10:
                energy = max(1, min(10, energy)) 
        except (ValueError, TypeError):
            energy = 5 
        params['energy_level'] = energy
        
        
        params['mood'] = params.get('mood') or "neutral"
        params['genre'] = params.get('genre') or "ambient"
        
        
        return params
    
    def _fallback_parsing(self, user_text: str) -> dict:
        """Basic keyword-based parsing when the LLM is unavailable."""
        
        print(f"LLM failed. Using fallback parser for request: '{user_text}'")
        text_lower = user_text.lower()
        
        
        mood = "neutral"
        if "happy" in text_lower or "birthday" in text_lower:
            mood = "happy"
        elif "sad" in text_lower or "breakup" in text_lower:
            mood = "sad"
        elif "calm" in text_lower or "meditation" in text_lower or "study" in text_lower:
            mood = "calm"

        energy_level = 5
        if "workout" in text_lower or "energetic" in text_lower:
            energy_level = 8
        elif "sleep" in text_lower or "meditation" in text_lower:
            energy_level = 2
            
            
        return {
            "mood": mood,
            "energy_level": energy_level,
            "genre": "ambient", 
            "tempo": "medium",
            "instruments": "synthesizer, pad",
            "context": "fallback music"
        }

if __name__ == "__main__":
    
    try:
        processor = InputProcessor()
       
        test_cases = [
            
            "I need energetic music for my workout.",
            "Something calming for meditation.",
            "Happy birthday party music!",
            "Sad breakup song",
            "Focus music for studying",
            "Play some rock music for a road trip.",
            "I want a slow, romantic piano melody.",
            "Make a fast, fun disco beat.",
            "A mysterious, dark ambient soundscape.",
            "Cheerful music for cleaning the house."
        ]
        
        print("\n--- Running LLM Input Processor Tests ---")
        for i, prompt in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Prompt: **{prompt}**")
            output = processor.process(prompt)
            print("Extracted Parameters:")
            
            
            print(json.dumps(output, indent=2))
            
    except ValueError as e:
        print(f"\n--- SETUP ERROR ---")
        print(e)
        print("Please ensure your OPENAI_API_KEY is in a .env file.")
        
    except Exception as e:
        print(f"\n--- GENERAL ERROR ---")
        print(f"An unexpected error occurred: {e}") 