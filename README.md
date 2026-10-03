# Melod AI 🎵
**Where Your Mind Meets Music — Through AI.**

Melod AI is a CPU-optimized, generative AI web application that translates text prompts into original audio tracks. Built entirely with Python and Streamlit, the application leverages Hugging Face's `musicgen-small` transformer model to synthesize music locally, complete with dynamic waveform visualizations and a responsive, custom dark-mode interface.

![Melod AI Dashboard](MelodAI%20Screenshots/Screenshot%202026-10-03%20150738.png)
*(Note: Replace `your_main_screenshot_name.png` with the actual filename of your best screenshot)*

## 🚀 Engineering Highlights & Optimizations

This project was engineered to run efficiently on standard local hardware, overcoming the typical CPU and memory bottlenecks associated with large language models and audio processing:

* **Persistent Resource Caching:** Implemented `@st.cache_resource` for the core machine learning pipeline (`InputProcessor`, `PromptEnhancer`, `Advance_Generator`). This secures multi-gigabyte transformer models persistently in memory, eliminating redundant load times during user interactions.
* **CPU-Constrained Inference:** Hardcoded the architecture to utilize `facebook/musicgen-small` with a strict single-track generation limit (num_variations=1) and an 8-second default duration, drastically reducing local CPU inference time.
* **Robust File & State Management:** Engineered a resilient frontend that actively prevents thread-crashing (e.g., `MediaFileStorageError`) by utilizing strict `os.path.exists()` validations before invoking media rendering or file deletion.
* **Memory Leak Prevention:** Refactored audio visualization pipelines (`librosa` and `matplotlib`) to explicitly close active plot figures (`plt.close(fig)`), resolving runtime memory leak warnings inherent to Streamlit's stateless execution model.
* **Cross-Platform Data Integrity:** Resolved local Windows Mojibake serialization bugs by enforcing strict `UTF-8` encoding protocols for reading and writing to the local JSON database (`melodai_history.json`).

## 🛠️ Tech Stack

* **Frontend & Backend Framework:** Streamlit (Python)
* **Generative AI Model:** `facebook/musicgen-small` (Hugging Face Transformers)
* **Audio Processing:** Librosa, SciPy
* **Data Visualization:** Matplotlib
* **Local Storage:** JSON Document Store

## ⚙️ Installation & Usage

To run this application locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NovaKrishnam/Melod_AI.git](https://github.com/NovaKrishnam/Melod_AI.git)
   cd Melod_AI