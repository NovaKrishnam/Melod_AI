import os
import json
import hashlib
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

class CacheManager:
    """
    Implements Task 3.5: Intelligent caching system for MelodAI.
    Features: TTL expiration, Exporting, and Hit analytics.
    """
    def __init__(self, cache_dir: str = 'audio_cache', max_files: int = 50, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_files = max_files
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.hits = 0
        self.misses = 0
        self.cache_registry = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_registry, f, indent=4)
        except Exception as e:
            print(f"CACHE ERROR: {e}")

    def get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Creates unique MD5 hash for prompt + parameters."""
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{prompt.lower().strip()}{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves results with 1-hour TTL check."""
        if cache_key in self.cache_registry:
            entry = self.cache_registry[cache_key]
            
            # --- TASK 3.5: 1-HOUR TTL CHECK ---
            # If older than 3600 seconds, call delete() and return None
            if time.time() - entry['timestamp'] > 3600:
                self.delete(cache_key) # This now works!
                self.misses += 1
                return None

            results = entry.get('results', [])
            valid_results = []
            for r in results:
                file_path = Path(r.get('file_path', ''))
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        r['audio_bytes'] = f.read()
                    valid_results.append(r)
            
            if valid_results:
                self.cache_registry[cache_key]['last_accessed'] = time.time()
                self._save_metadata()
                self.hits += 1
                return valid_results
        
        self.misses += 1
        return None

    def set(self, cache_key: str, results: List[Dict[str, Any]]):
        """Stores results and enforces LRU limits."""
        serializable_results = []
        for r in results:
            orig_path = Path(r['file_path'])
            cached_path = self.cache_dir / f"{cache_key}_{r['variation_id']}.mp3"
            shutil.copy(orig_path, cached_path)
            
            clean_item = r.copy()
            clean_item['file_path'] = str(cached_path)
            clean_item.pop('audio_bytes', None) # CRITICAL: Remove bytes for JSON
            serializable_results.append(clean_item)

        self.cache_registry[cache_key] = {
            "results": serializable_results,
            "timestamp": time.time(),
            "last_accessed": time.time(),
            "size": sum(os.path.getsize(Path(r['file_path'])) for r in serializable_results)
        }
        self._enforce_limits()
        self._save_metadata()

    def _enforce_limits(self):
        """Task 3.5: Stay under 50 files or 500MB."""
        while len(self.cache_registry) > self.max_files:
            self._evict_oldest()
            
        current_size = sum(item['size'] for item in self.cache_registry.values())
        while current_size > self.max_size_bytes and self.cache_registry:
            self._evict_oldest()

    def _evict_oldest(self):
        """Removes the least recently accessed item."""
        if not self.cache_registry: return
        oldest_key = min(self.cache_registry, key=lambda k: self.cache_registry[k]['last_accessed'])
        self.delete(oldest_key)

    def delete(self, cache_key: str):
        """
        NEW: Task 3.5 Cleanup logic.
        Permanently removes files from disk and registry.
        """
        if cache_key in self.cache_registry:
            # Delete physical audio files associated with this key
            for r in self.cache_registry[cache_key].get('results', []):
                p = Path(r.get('file_path', ''))
                if p.exists():
                    try:
                        os.remove(p)
                    except Exception as e:
                        print(f"Error deleting cached file: {e}")
            
            # Remove from metadata registry
            del self.cache_registry[cache_key]
            self._save_metadata()

    def clear_cache(self):
        """Clears all metadata and physical files."""
        for folder, subs, files in os.walk(self.cache_dir):
            for f in files:
                if f != "cache_metadata.json": 
                    try: os.remove(os.path.join(folder, f))
                    except: pass
        self.cache_registry = {}
        self._save_metadata()
        self.hits = 0
        self.misses = 0

    def get_most_cached_moods(self) -> str:
        """Analytics for Statistics tab."""
        if not self.cache_registry: return "None"
        moods = []
        for item in self.cache_registry.values():
            if item.get('results') and 'extracted_parameters' in item['results'][0]:
                moods.append(item['results'][0]['extracted_parameters'].get('mood', 'N/A'))
        
        if not moods: return "None"
        from collections import Counter
        return Counter(moods).most_common(1)[0][0]

    def get_stats(self) -> Dict[str, Any]:
        """Calculates dashboard metrics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        storage_used = sum(item['size'] for item in self.cache_registry.values()) / (1024 * 1024)
        return {
            "hit_rate": round(hit_rate, 2),
            "storage_used_mb": round(storage_used, 2),
            "total_files": len(self.cache_registry)
        }