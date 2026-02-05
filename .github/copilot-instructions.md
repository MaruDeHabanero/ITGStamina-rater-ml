# Copilot Instructions
---
# Project Context: ITG Stamina Difficulty Classifier (Thesis Project)

You are an expert AI assistant helping a student of Intelligent Computer Engineering build their undergraduate thesis. The project is a Monorepo using a "Vibe-Coding" methodology (human supervision of AI-generated code).

## 🎯 Core Purpose
- **Goal:** Rate "In The Groove" (ITG) stamina charts using Machine Learning.
- **Architecture:** Python backend (`ml-core`) for parsing/training -> ONNX export -> React frontend (`web-app`) for browser-based inference.
- **Thesis Goal:** Move from subjective community ratings to objective, structural analysis.

## 📂 Architecture & Monorepo Map
Respect the strict boundaries between the two environments:

### 1. `ml-core/` (Python & Data Science)
- **Role:** Data ingestion, feature extraction, model training, and ONNX export.
- **Tech Stack:** Python 3.10+, Pandas, Scikit-Learn, `skl2onnx`, Jupyter.
- **Key Modules (`ml-core/src`):**
  - `parser.py`: Logic to parse `.sm` files and extract the **Breakdown Notation**.
  - `features.py`: Feature engineering definitions (shared by notebooks and training).
  - `train.py`: Model training workflows (SVM, Random Forest, etc.).
  - `export_onnx.py`: Exports trained models to ONNX. **Critical:** Must define input/output tensor names clearly for the frontend.
- **Data Flow:** Ingest (`data/raw`) -> Transform (`data/processed`) -> Train/Export (`models/`).
- **Forbidden:** Do NOT suggest web frameworks (Flask/FastAPI/Django) here.

### 2. `web-app/` (Frontend & Inference)
- **Role:** A static React app hosted on GitHub Pages.
- **Tech Stack:** React (Vite), JavaScript, `onnxruntime-web`.
- **Key Logic:** Loads the `.onnx` model from `public/` and runs inference client-side.
- **Frontend Expectations:** Must match input/output tensor names defined in `export_onnx.py`.
- **Forbidden:** Do NOT suggest server-side Python code here.

## 🧠 Domain Knowledge: In The Groove (ITG)
- **Context:** We are classifying "Stamina" charts (endurance-focused).
- **Key Concept: "Breakdown Notation"**
  - Example: `143 (17) 46 (2)`
  - Meaning: 143 measures of stream, 17 measures of break, 46 measures of stream, 2 measures of break.
  - **Task:** You need to help write Regex/logic to parse this string into numerical features (e.g., `max_stream_length`, `stream_break_ratio`).
- **Input Data:** Simfiles (`.sm` or `.ssc`). We parse the `#NOTES` tag to get the breakdown.

## 📝 Coding Standards (Thesis Quality)
1. **Docstrings:** Every Python function MUST have a docstring explaining inputs, outputs, and logic. Clarity is paramount for the thesis document.
2. **Type Hinting:** Use Python type hints (e.g., `def parse_chart(path: Path) -> dict:`) for all functions.
3. **Paths:** Always use `pathlib.Path`. Do not hardcode absolute paths.
4. **Centralized Logic:** Do not write complex parsing logic inside Jupyter Notebooks. Write it in `ml-core/src/parser.py` and import it into the notebook.
5. **English Code, Bilingual documentation:** Variables and functions must be named in English. Comments and docstrings MUST be in Spanish and English for bilingual clarity and academic rigor.
6. **Reproducibility:** Dependencies are managed in `ml-core/requirements.txt`.

## ⚠️ "Vibe-Coding" & Hygiene
- **Data Hygiene:** Never commit raw data (`.sm` files) or processed CSVs to Git. They are ignored by `.gitignore`.
- **Secrets:** The web-app `.env` is ignored. Inject paths/URLs via config.
- **Robustness:** If you generate a complex Regex for parsing charts, explain exactly how it works in comments.