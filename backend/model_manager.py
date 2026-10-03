import torch
from typing import Dict, List

class ModelManager:
    """
    Manages multiple MusicGen model variants for the MelodAI backend.
    """
    def __init__(self):
        self.initialized_variants: Dict[str, bool] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_variants = [
            'facebook/musicgen-small',
            'facebook/musicgen-medium',
            'facebook/musicgen-large',
            'facebook/musicgen-melody'
        ]

    def load_model_variant(self, variant_name: str) -> str:
        if variant_name not in self.initialized_variants:
            print(f"--- Initializing MusicGenerator: {variant_name} on {self.device} ---")
            self.initialized_variants[variant_name] = True
        return variant_name

    def select_model_auto(self, user_selection: str, duration: float) -> str:
        # Task 3.3 Logic: Fallback to 'small' if duration > 30s to ensure stability
        if duration > 30.0 and "small" not in user_selection:
            print(f"AUTO-FALLBACK: Switching to 'small' model for {duration}s track.")
            return "facebook/musicgen-small"
        return user_selection if user_selection in self.supported_variants else "facebook/musicgen-small"