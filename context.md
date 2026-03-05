# ITGStamina-rater-ml — Comprehensive Project Context

> Generated 2026-03-04. Covers every source file, script, notebook, dataset, and report in the monorepo.

---

## 1. Project Overview

**Goal:** Classify the difficulty of "In The Groove" (ITG) stamina charts using Machine Learning, moving from subjective community ratings to objective, structural analysis.

**Architecture:** Python backend (`ml-core`) for parsing/training → ONNX export → React frontend (`web-app`) for browser-based inference via `onnxruntime-web`.

**Thesis context:** Undergraduate thesis for Intelligent Computer Engineering. "Vibe-Coding" methodology (human supervision of AI-generated code).

**Prior work / inspirations:**
- Eryk Banatt: [itsa17](https://github.com/ambisinister/itsa17) — Auto-Rating ITG Stamina Charts with ML
- som1sezhi: [itg-difficulty-predictor](https://github.com/som1sezhi/itg-difficulty-predictor/)

**License note:** Stream-detection logic is a GPLv3 port from Simply-Love-SM5 (2020, Simply Love Team). Modifications: Lua → Python 3.10+, 1-based → 0-based indexing, integration with scikit-learn pipeline.

---

## 2. Monorepo Structure

```
ITGStamina-rater-ml/
├── .github/copilot-instructions.md   # AI assistant rules
├── README.md                         # Bilingual project overview
├── TODO.md                           # Current task list
├── ml-core/
│   ├── requirements.txt              # Python dependencies
│   ├── src/
│   │   ├── parser.py                 # .sm file parser
│   │   └── features.py              # Feature engineering (ported Simply Love logic)
│   ├── scripts/
│   │   ├── build_dataset.py          # Batch .sm → CSV pipeline
│   │   ├── cleanup_raw.py            # Remove non-chart files
│   │   └── cleanup_unlocks.py        # Remove tech/non-stamina unlocks
│   ├── notebooks/
│   │   ├── 01_feature_validation.ipynb
│   │   ├── 02_stamina_rpg7_comparison.ipynb
│   │   └── 03_exploratory_data_analysis.ipynb
│   ├── data/
│   │   ├── raw/                      # .sm packs (gitignored)
│   │   └── processed/
│   │       ├── stamina_dataset.csv   # Training CSV (gitignored)
│   │       ├── cleanup_report.txt
│   │       └── remove_tech_report.txt
│   └── models/                       # Trained/exported models (empty so far)
└── web-app/                          # React + Vite + onnxruntime-web (stub)
```

---

## 3. Dependencies (`ml-core/requirements.txt`)

```
pandas
scikit-learn
skl2onnx
jupyter
typing_extensions
tqdm
matplotlib
numpy
seaborn
```

---

## 4. Source Modules

### 4.1 `ml-core/src/parser.py` — Simfile Parser

**Purpose:** Parse StepMania `.sm` files into per-measure note density arrays plus metadata (BPM, difficulty block).

#### Key Types

| Type | Description |
|---|---|
| `NotesData` (TypedDict) | `display_bpm` (float\|None), `timing_bpm` (float\|None), `block` (int — classification target), `notes_per_measure` (List[int]) |

#### Constants & Configuration

| Constant | Value | Meaning |
|---|---|---|
| `_NOTE_SYMBOLS` | `{"0","1","2","3","4","M"}` | Valid characters in a note row |
| `_NOTE_HEADS` | `{"1","2","4"}` | Characters counted as actual notes (tap, hold head, roll head) |
| `_DIFFICULTY_PRIORITY` | `{"challenge": 2, "hard": 1}` | Chart selection priority |
| `_VALID_SUBDIVISIONS` | `[16, 20, 24, 32]` | Valid ITG stamina subdivisions |
| `_TITLE_BPM_RE` | `r"\[\d+\]\s*\[(\d+(?:\.\d+)?)\]"` | Regex to extract BPM from `#TITLE` |

#### Key Functions

| Function | Signature | Description |
|---|---|---|
| `_load_clean_lines` | `(file_path: Path) → List[str]` | Load .sm file, strip `//` comment lines |
| `_extract_tag_payload` | `(lines, tag) → Optional[str]` | Extract `#TAG:...;` payload (handles multiline) |
| `_extract_timing_bpm` | `(lines) → Optional[float]` | Mode of `#BPMS` values (actual engine BPM) |
| `_extract_title_bpm` | `(lines) → Optional[float]` | BPM from `#TITLE:[block] [bpm] name` (community-perceived eBPM) |
| `_extract_charts` | `(lines) → List[dict]` | Extract all `#NOTES` blocks with metadata |
| `_select_hardest_chart` | `(charts) → dict` | Pick hardest dance-single (Challenge > Hard) |
| `_count_notes_per_measure` | `(note_lines) → List[int]` | Count note heads per measure |
| `_infer_subdivision` | `(notes_per_measure) → int` | Mode of stream measures (≥12 notes) → nearest valid subdivision |
| `parse_sm_chart_with_meta` | `(file_path: Path) → Tuple[NotesData, int]` | **Main entry point.** Returns NotesData + subdivision |
| `parse_sm_chart` | `(file_path: Path) → NotesData` | Convenience wrapper (ignores subdivision) |

#### BPM Duality Concept

- **`display_bpm`:** From `#TITLE` — the community-facing speed (e.g. 260 for a 32nd chart at base 130 BPM). This is the **authoritative speed reference**.
- **`timing_bpm`:** From `#BPMS` (mode) — the actual engine BPM. Used only for NPS/eBPM calculations. For 24th/32nd charts, differs from display_bpm (e.g. Flaklypa: timing=130, display=260).

#### Subdivision Detection Logic

1. Filter measures with ≥12 notes (stream-like)
2. Compute mode of those note counts
3. Snap to nearest valid subdivision: 16, 20, 24, or 32

---

### 4.2 `ml-core/src/features.py` — Feature Engineering

**Purpose:** Detect stream/break segments, compute breakdown metrics, calculate eBPM profiles, detect bursts, and generate breakdown notation strings. Core logic is a direct Python port of Simply Love SM5's Lua `GenerateBreakdownText` / `GetMeasureInfo`.

#### Constants

| Constant | Value | Meaning |
|---|---|---|
| `STREAM_THRESHOLD` | `16` | Default minimum notes-per-measure to classify as stream |

#### Threshold Cascade (Stream Detection)

The `get_stream_sequences` function tries thresholds in descending order to accommodate different subdivisions:

| Threshold | Multiplier | Use Case |
|---|---|---|
| 32 | 2.0× | 32nd charts |
| 24 | 1.5× | 24th charts |
| 20 | 1.25× | 20th charts |
| 16 | 1.0× | Standard 16th charts |

A candidate is accepted only if scaled stream density ≥ 0.2. Lengths are scaled by `multiplier` to normalize to 16th-equivalent measures.

#### Key Functions

| Function | Signature | Description |
|---|---|---|
| `_build_segments` | `(notes_per_measure, threshold) → List[StreamSegment]` | Raw stream/break segment builder. Strips leading/trailing breaks. `stream_sequence_threshold=1`, `break_sequence_threshold=2` |
| `_compute_density` | `(segments, multiplier) → float` | `scaled_stream / scaled_total` |
| `get_stream_sequences` | `(notes_per_measure, subdivision?) → List[StreamSegment]` | **Main stream detection.** Returns segments with `start`, `end`, `is_break`, `length`, `scaled_length`, `threshold`, `multiplier` |
| `calculate_ebpm_profile` | `(notes_data) → Dict` | Per-measure NPS & eBPM arrays using `timing_bpm`. `measure_seconds = 240.0 / timing_bpm`, `nps = count / measure_seconds`, `ebpm = nps * 15.0` |
| `calculate_breakdown_metrics` | `(notes_input, subdivision?) → Dict` | **Core feature extraction.** Returns all ML features (see below) |
| `detect_bursts` | `(notes_data, sequences, nps_ratio=0.5) → List[BurstInfo]` | Detect fast arrow clusters inside break segments. Threshold: 50% of full-stream NPS |
| `summarize_bursts` | `(bursts) → Dict` | Aggregate burst metrics |
| `generate_breakdown_string` | `(notes_per_measure, subdivision?) → str` | Human-readable breakdown, e.g. `"20 (2) 30 (8) 16"` |

#### ML Features (from `calculate_breakdown_metrics`)

| Feature | Type | Description |
|---|---|---|
| `total_stream_length` | int | Sum of all stream segment scaled lengths |
| `max_stream_length` | int | Longest continuous stream segment (scaled) |
| `break_count` | int | Number of break segments |
| `stream_break_ratio` | float | `total_stream / total_break`. **0.0 = pure stream (no breaks)** |
| `average_nps` | float | Mean NPS across all measures (using `timing_bpm`) |
| `ebpm` | float\|None | `PeakNPS × 15.0` — peak per-measure NPS scaled to BPM equivalent. Ported from Simply Love's `peakNPS` accumulator |

#### eBPM Calculation (Ported from Simply Love)

```
measure_seconds = 240.0 / timing_bpm
peak_nps = max(count / measure_seconds for count in notes_per_measure)
ebpm = peak_nps × 15.0
```

Validation examples:
- Flaklypa (32nds at 130 BPM base) → timing=130, max_notes=32 → PeakNPS≈17.33 → eBPM=260 ✓
- Pure 16ths at 200 BPM → timing=200, max_notes=16 → PeakNPS≈13.33 → eBPM=200 ✓

#### Burst Detection Logic

- Only within break segments
- Threshold: `nps_ratio × full_stream_nps` (default 50% of 16-notes-per-measure NPS)
- Groups consecutive measures above threshold
- Returns per-burst: `start`, `end`, `length`, `total_notes`, `avg_nps`, `peak_nps`, `avg_ebpm`, `peak_ebpm`

---

## 5. Scripts

### 5.1 `ml-core/scripts/build_dataset.py` — Dataset Builder

**Purpose:** Batch-process all `.sm` files under `data/raw/`, extract features, deduplicate, and emit `data/processed/stamina_dataset.csv`.

#### Pipeline Flow

1. Recursively find all `.sm` files, sorted alphabetically by path
2. Parse each file with `parse_sm_chart_with_meta`
3. Extract features via `calculate_breakdown_metrics`
4. Generate breakdown string via `generate_breakdown_string`
5. Compute MD5 hash of `notes_per_measure` list for deduplication
6. On hash collision: later entry (higher-numbered pack) overwrites → handles re-rates across pack versions
7. Quality filter: discard charts with `total_stream_length < MIN_STREAM_LENGTH` (4 measures)
8. Truncate MD5 to 8-char `chart_id`
9. Reorder columns and save to CSV

#### Constants

| Constant | Value | Description |
|---|---|---|
| `RAW_DIR` | `ml-core/data/raw` | Default input directory |
| `OUTPUT_PATH` | `ml-core/data/processed/stamina_dataset.csv` | Default output path |
| `MIN_STREAM_LENGTH` | `4` | QA threshold — charts below this are discarded as parser errors |

#### Key Functions

| Function | Signature | Description |
|---|---|---|
| `_md5_of_notes` | `(notes_per_measure) → str` | MD5 hex-digest of note density list |
| `process_file` | `(file_path, raw_dir) → Optional[Dict]` | Parse one .sm → flat feature record |
| `build_dataset` | `(raw_dir) → pd.DataFrame` | Full pipeline: traverse, parse, deduplicate, filter, reorder |
| `main` | `() → None` | CLI entry point with argparse |

#### Column Order in Output

```
chart_id | difficulty | breakdown | source_file | display_bpm | ebpm |
total_stream_length | max_stream_length | break_count | stream_break_ratio | average_nps
```

---

### 5.2 `ml-core/scripts/cleanup_raw.py` — Raw Data Cleanup

**Purpose:** Remove every file that is NOT a `.sm` or `.ssc` chart from `data/raw/` (audio files, images, `Pack.ini`, `.ogg`, `.png`, etc.).

#### Configuration

| Constant | Value |
|---|---|
| `ALLOWED_EXTENSIONS` | `{".sm", ".ssc"}` |

#### Key Functions

| Function | Description |
|---|---|
| `list_extra_files(raw_root)` | Collect non-chart files recursively |
| `delete_files(paths)` | Delete files |
| `collect_file_stats(root)` | Count files + total bytes |
| `write_report(...)` | Persist cleanup changelog |

#### CLI

- `--raw-root` — custom raw data root
- `--dry-run` — preview without deleting
- `--report-path` — output report location

---

### 5.3 `ml-core/scripts/cleanup_unlocks.py` — Tech/Non-Stamina Unlocks Removal

**Purpose:** Remove non-stamina "tech" charts from `"- Unlocks"` pack subfolders by inspecting the `#BANNER` metadata in `.sm` files.

#### Configuration

```python
TECH_BANNERS = {
    "srpg5zdprtunlockbn.png",
    "srpg6zdprtunlockbn.png",
    "srpg7bdprtunlockbn.png",
    "srpg8bdprtunlockbn.png",
}
```

These banner filenames identify tech/non-stamina charts in Stamina RPG Unlocks packs (5–8).

#### Key Functions

| Function | Description |
|---|---|
| `find_sm_files(chart_folder)` | Find `.sm` files in a chart folder |
| `read_banner(sm_path)` | Read `#BANNER` value from `.sm` |
| `find_tech_candidates(root)` | Scan `*- Unlocks` dirs for tech charts |
| `delete_candidates(candidates, apply)` | Delete or dry-run |

---

## 6. Data Reports

### 6.1 `cleanup_report.txt` — Raw Data Cleanup Results

```
Raw root: /home/maru/.../ml-core/data/raw
Space before deletion: 59.88 GB (64,299,745,934 bytes)
Space after deletion:  174.12 MB (182,578,256 bytes)
Total files before:    14,692
Files deleted:         9,079
Total files after:     5,613
```

**Impact:** Reduced raw data folder from ~60 GB to ~174 MB by removing audio files (`.ogg`), images (`.png`), `Pack.ini`, `Thumbs.db`, and other non-chart files. 9,079 files deleted across all packs.

### 6.2 `remove_tech_report.txt` — Tech Unlocks Removal Results

```
Charts marcados (tech): 206
Espacio estimado: 4.10 MiB
```

**Impact:** 206 tech/non-stamina chart folders identified across Stamina RPG 5–8 Unlocks packs via banner matching. These charts use patterns not relevant to stamina classification (crossovers, footswitches, etc.).

---

## 7. Dataset: `stamina_dataset.csv`

### 7.1 Shape

- **2,851 rows** × **11 columns**
- No null values

### 7.2 Column Definitions

| Column | dtype | Description |
|---|---|---|
| `chart_id` | str | 8-char truncated MD5 of note density (dedup key) |
| `difficulty` | int64 | **Target variable.** Community-assigned difficulty block (11–43) |
| `breakdown` | str | Human-readable breakdown notation, e.g. `"143 (17) 46 (2)"` |
| `source_file` | str | Relative path from `data/raw/` for traceability |
| `display_bpm` | float64 | Community-facing BPM from `#TITLE` |
| `ebpm` | float64 | Effective BPM (`PeakNPS × 15`) |
| `total_stream_length` | int64 | Total stream measures (scaled to 16th equivalents) |
| `max_stream_length` | int64 | Longest continuous stream segment |
| `break_count` | int64 | Number of break segments |
| `stream_break_ratio` | float64 | Stream/break ratio (0.0 = pure stream) |
| `average_nps` | float64 | Mean notes-per-second across all measures |

### 7.3 Target Distribution

| Level | Count | | Level | Count | | Level | Count |
|---:|---:|---|---:|---:|---|---:|---:|
| 11 | 58 | | 20 | 150 | | 29 | 62 |
| 12 | 163 | | 21 | 150 | | 30 | 44 |
| 13 | 179 | | 22 | 144 | | 31 | 35 |
| 14 | 185 | | 23 | 125 | | 32 | 25 |
| 15 | 186 | | 24 | 119 | | 33 | 21 |
| 16 | 187 | | 25 | 120 | | 34 | 20 |
| 17 | 171 | | 26 | 117 | | 35 | 20 |
| 18 | 181 | | 27 | 95 | | 36 | 17 |
| 19 | 141 | | 28 | 81 | | 37–43 | 55 total |

**Range:** 11–43. **Mean:** 20.31. **Median:** 19.0. Roughly balanced in the 12–22 range (~150–187 charts each), tapering off above 27.

### 7.4 Feature Statistics

| Stat | difficulty | display_bpm | ebpm | total_stream | max_stream | break_count | stream_break_ratio | avg_nps |
|---|---|---|---|---|---|---|---|---|
| mean | 20.31 | 228.5 | 228.1 | 259.3 | 68.2 | 6.4 | 8.56 | 11.99 |
| std | 6.36 | 67.1 | 66.1 | 393.6 | 85.9 | 12.8 | 17.83 | 3.67 |
| min | 11 | 120 | 59.4 | 14 | 1 | 0 | 0.0 | 2.46 |
| 25% | 15 | 175.5 | 177.8 | 90 | 28 | 2 | 1.90 | 9.31 |
| 50% | 19 | 218 | 218 | 145 | 47 | 3 | 3.88 | 11.56 |
| 75% | 24 | 270 | 270 | 252 | 82 | 6 | 7.89 | 14.01 |
| max | 43 | 480 | 480 | 6,046 | 2,512 | 273 | 369.5 | 27.32 |

**Key observations:**
- `display_bpm` and `ebpm` are very similar (mean 228 vs 228) — most charts are standard 16ths
- `total_stream_length` has extreme right skew (max 6,046 vs median 145) — long marathon charts
- `stream_break_ratio` = 0.0 means pure stream (no breaks at all)

---

## 8. Raw Data Sources

### Packs Included

| Series | Packs | Tiers |
|---|---|---|
| **East Coast Stamina (ECS)** | ECS 10, 11, 12, 13, 14 | Lower, Mid, Upper, Speed |
| **Stamina RPG (SRPG)** | SRPG 5, 6, 7, 8, 9 | Main + Unlocks |

The `"- Unlocks"` subfolders contain bonus charts, some of which are tech (non-stamina) and are filtered by `cleanup_unlocks.py`.

### Deduplication Strategy

Files are sorted alphabetically by path → higher-numbered packs processed last → on MD5 collision the newer re-rate wins. This handles charts that appear in multiple packs with different difficulty ratings.

---

## 9. Notebooks

### 9.1 `01_feature_validation.ipynb` — Single Chart Feature Validation

**Purpose:** Validate the parser + feature engineering pipeline on a single chart (`Stamina RPG 6/[23] Cycle Hit/Cycle Hit.sm`).

**Sections:**
1. Load and parse a single `.sm` file
2. Plot per-measure NPS density with reference lines (Display BPM NPS, eBPM NPS)
3. Compute and display stream/break sequences
4. Generate breakdown string and summary metrics
5. Print final analysis result

**Key outputs:** Visual confirmation that the parser correctly identifies stream zones, that subdivision detection works, and that metrics match manual inspection.

---

### 9.2 `02_stamina_rpg7_comparison.ipynb` — Same-Level Chart Comparison

**Purpose:** Compare three charts from Stamina RPG 7, all rated difficulty 16, to demonstrate how structural features differentiate charts of the same difficulty level.

**Charts compared:**
| Chart | File |
|---|---|
| Plasticworld (Easy) | `plasticworld.sm` |
| Holding Out For A Hero | `Holding Out For A Hero.sm` |
| ITC JAMS Vol 4 | `ITCJ4.sm` |

**Sections:**
1. Setup and parse all three charts
2. Overlay NPS density plot (all three on one axis)
3. Individual NPS profiles (3 subplots)
4. Comparative feature table (DataFrame)
5. Breakdown notation for each
6. Bar chart comparison: total stream, max stream, break count, stream/break ratio
7. Box plot: distribution of stream segment lengths per chart
8. Per-measure eBPM profile with shaded stream zones
9. Final text summary

**Key findings:** Three charts at the same difficulty rating (16) show very different structural profiles — different BPMs, breakdown patterns, stream lengths, and break distributions. This motivates using structural features beyond just BPM.

---

### 9.3 `03_exploratory_data_analysis.ipynb` — Full Dataset EDA

**Purpose:** Exploratory analysis of the full `stamina_dataset.csv` (2,851 charts) to guide feature selection before model training.

**Sections:**

#### Section 1: Load Data
- Shape: 2,851 × 11, no nulls

#### Section 2: Target Distribution
- Histogram of difficulty levels
- Roughly balanced 12–22, tapers above 27

#### Section 3: Correlation Matrix
- Pearson correlation heatmap over all numeric features
- Correlations with `difficulty` sorted by |r|

#### Section 4: Feature vs Target Scatter Plots
- `total_stream_length` vs difficulty (with jitter)
- `average_nps` vs difficulty (with jitter)

#### Section 5: Feature–Target Correlation Ranking
- Pearson + Spearman correlation bar chart
- Comparison shows both linear and monotonic relationships

#### Section 6: Violin Plots
- Top 4 features by |Spearman ρ| plotted as violin plots against difficulty

#### Section 7: BPM vs Breakdown — Incremental Predictive Value

**Key analysis.** Three complementary techniques:

##### 7.1 Partial Correlations (controlling for `display_bpm`)

$$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}$$

Key finding: `ebpm` drops to partial correlation ~0.06 after controlling for BPM → nearly redundant.

##### 7.2 R² and MAE Comparison (Linear Regression, 5-Fold CV)

| Model | R² (in-sample) | MAE (5-fold CV) | Exact match % | Within ±1 % |
|---|---|---|---|---|
| BPM only | ~0.93 | ~1.4 levels | — | ~59% |
| Breakdown only | ~0.07 | — | — | — |
| BPM + Breakdown | ~0.98 | ~0.8 levels | — | ~89% |

**MAE reduction:** ~44% by adding breakdown features.

##### 7.3 Where BPM Alone Fails

~16% of charts have BPM-only error ≥ 3 levels. These are **endurance marathons** at moderate BPMs:
- Average `total_stream_length` of big misses: ~575 (vs dataset avg ~259)
- These are precisely the charts where breakdown notation is essential

#### Key Takeaway (Section 7.4)

> **Thesis argument:** BPM sets the base difficulty range; breakdown notation features refine the prediction within that range. Both inputs are necessary — BPM without breakdown is imprecise; breakdown without BPM is nearly useless (R² = 0.07).

- `ebpm` is nearly redundant with `display_bpm` (partial r ≈ 0.06) → candidate for removal to reduce multicollinearity

---

## 10. Domain Knowledge: ITG Stamina

### Breakdown Notation

The central concept. Example: `143 (17) 46 (2)`

- `143` = 143 measures of continuous stream (arrows every 16th note, ~4 per beat)
- `(17)` = 17 measures of break (rest)
- `46` = 46 measures of stream
- `(2)` = 2 measures of break

### Stream vs Break

- **Stream:** Measures with ≥ `STREAM_THRESHOLD` (16) notes — continuous arrow patterns requiring sustained stamina
- **Break:** Measures below threshold — rest periods

### Subdivisions

| Subdivision | Notes/Measure | Description |
|---|---|---|
| 16 | 16 | Standard 16th notes — most common in stamina |
| 20 | 20 | 20th notes — rare |
| 24 | 24 | 24th notes — popular alternative |
| 32 | 32 | 32nd notes — very rare, effectively 2× the BPM |

### BPM Duality

- **Display BPM:** Community-perceived speed from `#TITLE` tag. E.g., a 32nd chart at base 130 BPM is labeled as 260 BPM.
- **Timing BPM:** Engine-actual BPM from `#BPMS` tag. Used internally for duration calculations.

### eBPM (Effective BPM)

`eBPM = PeakNPS × 15.0` — the BPM equivalent of the chart's densest measure. For standard 16th charts, eBPM ≈ display_bpm. For charts with harder sections (bursts, dense patterns), eBPM may exceed display_bpm.

---

## 11. TODO / Current Task List

From `TODO.md` (dated 12/02/26):
- Fix identification of 16th vs 24th charts
- Probably need to calculate max NPS for fair difficulty classification

---

## 12. Coding Standards

1. **Docstrings:** Every function has bilingual (ES/EN) docstrings
2. **Type hints:** All functions use Python type hints
3. **Paths:** Always `pathlib.Path`, no hardcoded absolute paths
4. **Centralized logic:** Parsing/feature logic in `src/`, not in notebooks
5. **English code, bilingual docs:** Variable names in English, comments/docstrings in both Spanish and English
6. **Data hygiene:** Raw `.sm` files and CSVs are gitignored
7. **Reproducibility:** Dependencies in `requirements.txt`

---

## 13. Web App (Stub)

`web-app/` is a planned React + Vite application for browser-based inference:
- Loads `.onnx` model from `public/`
- Uses `onnxruntime-web` for client-side inference
- Must match tensor names defined in (future) `export_onnx.py`
- No server-side Python — fully static, hosted on GitHub Pages

---

## 14. Data Pipeline Summary

```
Raw .sm files (ECS 10-14, SRPG 5-9)
    │
    ├─ cleanup_raw.py ──────── Remove non-chart files (59 GB → 174 MB)
    ├─ cleanup_unlocks.py ──── Remove 206 tech charts from Unlocks packs
    │
    ▼
Clean .sm files (~5,613 files)
    │
    ├─ parser.py ──────────── Parse notes, extract BPM, detect subdivision
    ├─ features.py ────────── Stream detection, breakdown metrics, eBPM
    │
    ▼
build_dataset.py
    │
    ├─ MD5 deduplication (newer pack wins)
    ├─ Quality filter (total_stream > 4)
    │
    ▼
stamina_dataset.csv (2,851 unique charts × 11 columns)
    │
    ▼
Notebooks (validation, comparison, EDA)
    │
    ▼
Model training (planned: SVM, Random Forest, Gradient Boosting)
    │
    ▼
ONNX export → web-app inference (planned)
```
