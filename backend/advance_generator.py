import os
from typing import Dict, Any, List, Tuple

# Import the necessary backend components
from .music_generator import MusicGenerator 
from .quality_scorer import QualityScorer 


class Advance_Generator:
    """
    Handles music generation with advanced features, primarily implementing 
    the QualityScorer auto-retry mechanism (Task 3.1) and Multi-Model support (Task 3.3).
    """
    def __init__(self):
        self.music_generator = MusicGenerator()
        self.quality_scorer = QualityScorer()

    def generate_music_with_quality_check(
        self, 
        prompt: str, 
        duration: int, 
        energy_level: int,
        model_name: str = "facebook/musicgen-small"  # <--- FIX: Added model_name argument
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generates music using the specified model, checks quality, and retries if necessary.
        """
        config = self.quality_scorer.config
        max_retries = config['max_retries']
        min_score = config['min_overall_score']
        
        all_results: List[Dict[str, Any]] = []
        
        for attempt in range(max_retries + 1):
            
            # 1. Generate the audio file using the core generator
            # PASS model_name to music_generator.generate
            audio_result = self.music_generator.generate(
                prompt=prompt,
                duration=duration,
                energy_level=energy_level,
                model_name=model_name  # <--- NEW: Ensuring model variant is used
            )
            
            audio_path = audio_result.get('file_path')
            
            # Handle critical generation failure
            if not audio_path or audio_result.get('status') != 'success':
                error_message = audio_result.get('message', 'Core Generation Failed')
                return {'status': 'failed', 'message': error_message}, {'overall_score': 0, 'error': error_message}

            # 2. Score the generated audio
            expected_params = {'duration': duration}
            score_report = self.quality_scorer.score_audio(audio_path, expected_params)
            overall_score = score_report.get('overall_score', 0)
            
            # Store attempt results
            all_results.append({
                'score': overall_score, 
                'report': score_report, 
                'result_dict': audio_result, 
                'attempt': attempt + 1
            })
            
            # 3. Check threshold
            if overall_score >= min_score:
                print(f"   ✅ Quality Check PASS on attempt {attempt + 1}. Score: {overall_score}.")
                audio_result['quality_score'] = overall_score
                return audio_result, score_report
            
            if attempt < max_retries:
                print(f"   ❌ Quality Check FAIL on attempt {attempt + 1}. Score: {overall_score}. Retrying...")
            
        # If max retries reached, return the track from the attempt with the highest score
        best_result = max(all_results, key=lambda x: x['score'])
        print(f"   ⚠️ Max retries reached. Returning best available track (Score: {best_result['score']}).")
        
        best_result['result_dict']['status'] = 'partial_success'
        best_result['result_dict']['quality_score'] = best_result['score']
        
        return best_result['result_dict'], best_result['report']