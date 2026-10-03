import sys
import os
import json
import logging
from typing import Dict, Any, List
import csv # <--- ADDED IMPORT

logging.basicConfig(level=logging.CRITICAL) 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from backend.input_processor import InputProcessor
from backend.prompt_enhancer import PromptEnhancer
from backend.advance_generator import Advance_Generator 

NUM_VARIATIONS = 3

try:
    
    INPUT_PROCESSOR = InputProcessor()
    PROMPT_ENHANCER = PromptEnhancer()
    ADVANCE_GENERATOR = Advance_Generator()
    
except Exception as e:
    
    print(f"\n CRITICAL INIT FAILURE: Could not load models or services. Error: {e}")
    sys.exit(1)


def generate_music_pipeline(user_input: str) -> List[Dict[str, Any]]:
    """
    Orchestrates the entire process: LLM -> Enhancer -> Advance Generator (with Quality Check).
    """
    results = []
    
    try:
        
        params = INPUT_PROCESSOR.process(user_input)
        params_dict = params if isinstance(params, dict) else params.__dict__
        
        enhanced_prompts = PROMPT_ENHANCER.enhance_prompt(params_dict, num_variations=NUM_VARIATIONS)

        
        for i, enhanced_prompt in enumerate(enhanced_prompts, 1):
            
            duration = params_dict.get('duration', 10) 
            energy_level = params_dict.get('energy_level', 5) 
            
            # --- CALL NEW ADVANCE GENERATOR METHOD ---
            audio_result, score_report = ADVANCE_GENERATOR.generate_music_with_quality_check(
                prompt=enhanced_prompt,
                duration=duration,
                energy_level=energy_level
            )
            # ------------------------------------------
            
            # Check for critical errors from the generation pipeline
            if audio_result.get('status') == 'failed':
                 raise Exception(audio_result.get('message', 'Generation Failed'))

            
            audio_result['enhanced_prompt'] = enhanced_prompt
            audio_result['variation_id'] = i
            audio_result['quality_score_report'] = score_report # Attach the full report
            results.append(audio_result)
            
            # Handle non-critical failure (if it was a partial_success/quality fail)
            if audio_result['status'] == 'critical_error':
                 raise Exception(audio_result.get('message', 'Critical Error during pipeline'))


    except Exception as e:
        return [{"status": "critical_error", "message": str(e), "user_input": user_input}]

    return results


# --- NEW FUNCTION TO SAVE REPORT (Needed for CSV creation) ---

def save_report_to_csv(all_generation_results: List[List[Dict[str, Any]]], filename: str = 'quality_scoring_report_task3_1.csv'):
    """Extracts and saves key quality metrics from the nested results structure to a CSV file."""
    
    fieldnames = [
        'Run_ID', 'Variation_ID', 'Original_Prompt', 'Requested_Duration', 
        'Final_Overall_Score', 'Final_Status', 'File_Path', 
        'Duration_Accuracy_Score', 'Audio_Quality_Score'
    ]
    
    report_data = []
    
    for run_id, prompt_results in enumerate(all_generation_results, 1):
        for result in prompt_results:
            score_report = result.get('quality_score_report', {})
            overall_score = score_report.get('overall_score', 0)
            
            # Use os.path.basename to keep the report clean
            file_path = os.path.basename(result.get('file_path', 'N/A')) 
            
            report_data.append({
                'Run_ID': run_id,
                'Variation_ID': result['variation_id'],
                'Original_Prompt': result.get('enhanced_prompt', 'N/A')[:50] + '...',
                'Requested_Duration': result.get('quality_score_report', {}).get('expected_params', {}).get('duration', 10),
                'Final_Overall_Score': overall_score,
                'Final_Status': 'PASS' if overall_score >= 65 else 'FAIL',
                'File_Path': file_path,
                'Duration_Accuracy_Score': score_report.get('duration_accuracy', 0),
                'Audio_Quality_Score': score_report.get('audio_quality', 0)
            })

    # Write to CSV
    # The file will be saved in the directory from which the script is executed (Project Root)
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_data)
        
        print(f"\n\n✅ DELIVERABLE GENERATED: The quality scoring report is saved to: {filename}")
        print(f"Total {len(report_data)} samples scored.")
    except Exception as e:
        print(f"\n\n❌ ERROR SAVING REPORT: Could not write CSV file. Error: {e}")


if __name__ == "__main__":
    
    
    test_prompts = [
        "A smooth, calming acoustic guitar melody for deep relaxation.", 
        "High-octane synthwave music with a driving beat for working out.", 
        "Something melancholy and slow with heavy strings for a sad, rainy evening.", 
        "Upbeat tropical house track with steel drums for a summer party.",
        "Epic cinematic orchestral music, high energy, like a movie trailer.", 
        "Quirky electronic music with unexpected sounds and a fast rhythm.", 
        "Funky 70s disco with a wah guitar and a classic bassline.", 
        "A neutral, medium tempo ambient loop for studying.", 
        "Aggressive industrial rock, dark mood, long duration (20 seconds).", 
        "A peaceful piece for yoga, very low energy and soft bells." 
    ]
    
    all_results_for_report = []
    
    print(" STARTING TASK 3.1: QUALITY SCORING EXECUTION")
    
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[RUN {i}/{len(test_prompts)}] Prompt: **{prompt[:50]}...**")
        
        results = generate_music_pipeline(prompt)
        all_results_for_report.append(results) # Collect all results for the final report
        
        
        status_line = f"  -> Flow: LLM Input -> Enhancer -> Advance Generator (x{NUM_VARIATIONS}, Max 2 Retries)"
        
        final_statuses = [r.get('status', 'error') for r in results]
        
        if all(s == 'success' for s in final_statuses):
              print(status_line + " | **ALL PASS (0 Retries)**")
        elif all(s in ('success', 'partial_success') for s in final_statuses):
              successful_files = [os.path.basename(r.get('file_path', '')) for r in results if r.get('file_path')]
              print(status_line + " | **COMPLETE (Mixed Quality)**")
              print(f"  Output Files Generated: {len(successful_files)}/{NUM_VARIATIONS}")
        else:
              print(status_line + " | **CRITICAL FAILURE**")
              print(f"  ❌ Error: {results[0].get('message', 'Unknown error')[:60]}...")
    
    
    # --- CALL THE REPORT FUNCTION (This is what creates the file) ---
    save_report_to_csv(all_results_for_report)
    
    print("\n FINAL EXECUTION SUMMARY COMPLETE. Task 3.1 Deliverable is ready.") 