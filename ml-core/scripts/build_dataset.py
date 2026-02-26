"""
English: Batch-process all .sm files in ml-core/data/raw/ and emit a training CSV.

Each file is parsed, features are extracted, and records are deduplicated by the
MD5 hash of the raw note density list. When two files share a hash the one processed
last (files are sorted alphabetically so higher-numbered packs win) overwrites the
earlier entry, handling re-rates across pack versions.

Output: ml-core/data/processed/stamina_dataset.csv

Español: Procesa en lote todos los archivos .sm en ml-core/data/raw/ y genera un CSV
de entrenamiento.

Cada archivo es parseado, se extraen sus características y los registros se deduplicán
por el hash MD5 de la densidad de notas cruda. Cuando dos archivos comparten hash el
procesado en último lugar (los archivos están ordenados alfabéticamente para que los
packs de numeración mayor ganen) sobreescribe la entrada anterior, manejando
re-rateos entre versiones de packs.

Salida: ml-core/data/processed/stamina_dataset.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ES: Añadir src/ al path para importar los módulos del proyecto.
# EN: Add src/ to path so project modules are importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ML_CORE_ROOT = _SCRIPT_DIR.parent
_SRC_PATH = _ML_CORE_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from features import calculate_breakdown_metrics, generate_breakdown_string  # noqa: E402
from parser import parse_sm_chart_with_meta  # noqa: E402

# ---------------------------------------------------------------------------
# ES: Rutas por defecto del proyecto.
# EN: Default project paths.
# ---------------------------------------------------------------------------
RAW_DIR: Path = _ML_CORE_ROOT / "data" / "raw"
OUTPUT_PATH: Path = _ML_CORE_ROOT / "data" / "processed" / "stamina_dataset.csv"

# ES: Umbral mínimo de medidas de stream para considerar un chart válido.
#     Charts por debajo de este valor son descartados como errores de parseo
#     (subdivisión mal detectada por bursts o notas especiales).
# EN: Minimum stream measure count to consider a chart valid.
#     Charts below this threshold are discarded as parse errors
#     (subdivision misdetected due to bursts or special notes).
MIN_STREAM_LENGTH: int = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _md5_of_notes(notes_per_measure: List[int]) -> str:
    """
    English: Compute the MD5 hex-digest of a note density list.

    The list is converted to its string representation before hashing so that
    identical note sequences always produce the same digest regardless of the
    source file path.

    Español: Calcula el hex-digest MD5 de una lista de densidades de notas.

    La lista se convierte a su representación de string antes de hashear para que
    secuencias de notas idénticas siempre produzcan el mismo digest sin importar
    la ruta del archivo fuente.

    Args:
        notes_per_measure: Lista de enteros con la cantidad de notas por compás.
                           / List of integers with note count per measure.

    Returns:
        Hex-digest MD5 de 32 caracteres. / 32-character MD5 hex-digest.
    """

    raw = str(notes_per_measure).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def process_file(file_path: Path, raw_dir: Path) -> Optional[Dict]:
    """
    English: Parse one .sm file and return a flat feature record, or None on error.

    Calls `parse_sm_chart_with_meta` to obtain the per-measure note density,
    Display BPM, numeric difficulty block, and detected subdivision. Then calls
    `calculate_breakdown_metrics` to compute all engineered features. The result
    is a single flat dict suitable for a DataFrame row.

    Español: Parsea un archivo .sm y devuelve un registro de características plano,
    o None en caso de error.

    Llama a `parse_sm_chart_with_meta` para obtener la densidad de notas por
    compás, el Display BPM, el bloque de dificultad numérico y la subdivisión
    detectada. Luego llama a `calculate_breakdown_metrics` para calcular todas
    las características diseñadas. El resultado es un dict plano apto para una
    fila de DataFrame.

    Args:
        file_path: Ruta absoluta al archivo .sm. / Absolute path to the .sm file.
        raw_dir: Raíz de los datos crudos (para la ruta relativa en source_file).
                 / Raw data root (used to derive the relative path in source_file).

    Returns:
        Diccionario con todas las características más `difficulty`, `breakdown`,
        `source_file` y `_notes_hash`, o None si el archivo no pudo parsearse.
        / Dict with all features plus `difficulty`, `breakdown`, `source_file`,
        and `_notes_hash`, or None if the file could not be parsed.
    """

    try:
        notes_data, subdivision = parse_sm_chart_with_meta(file_path)
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", file_path, exc)
        return None

    notes_per_measure: List[int] = notes_data["notes_per_measure"]
    difficulty: int = notes_data["block"]
    display_bpm: Optional[float] = notes_data.get("display_bpm")

    try:
        metrics = calculate_breakdown_metrics(notes_data, subdivision=subdivision)
        # ES: El breakdown es la representación canónica del chart, clave para la tesis.
        # EN: The breakdown is the canonical chart representation, key for the thesis.
        breakdown = generate_breakdown_string(notes_per_measure, subdivision=subdivision)
    except Exception as exc:
        logger.error("Feature extraction failed for '%s': %s", file_path, exc)
        return None

    # ES: Determinar la ruta relativa respecto a raw_dir para trazabilidad.
    # EN: Compute relative path from raw_dir for traceability.
    try:
        rel_path = str(file_path.relative_to(raw_dir))
    except ValueError:
        rel_path = str(file_path)

    notes_hash = _md5_of_notes(notes_per_measure)

    record: Dict = {
        "difficulty": difficulty,
        "breakdown": breakdown,
        "source_file": rel_path,
        "display_bpm": display_bpm,
        "_notes_hash": notes_hash,
        **metrics,
    }
    return record


def build_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    English: Traverse raw_dir recursively, parse every .sm file, deduplicate by
    MD5, and return a clean DataFrame of features.

    Files are sorted by path string before processing so that higher-numbered pack
    folders (e.g. 'ECS 14') are processed after lower-numbered ones ('ECS 10').
    On a hash collision the later entry overwrites the earlier one, meaning the
    most recent re-rate of a chart wins.

    Post-processing steps applied before returning:
    - MD5 hash truncated to 8 hex chars and saved as `chart_id` (first column)
    - Columns reordered to match community spreadsheet convention:
      chart_id | difficulty | source_file | display_bpm | ebpm |
      total_stream_length | max_stream_length | break_count |
      stream_break_ratio | average_nps

    Español: Recorre raw_dir de forma recursiva, parsea todos los archivos .sm,
    deduplica por MD5 y devuelve un DataFrame limpio de características.

    Los archivos se ordenan por ruta antes de procesarlos para que los packs de
    mayor numeración (p.ej. 'ECS 14') se procesen después de los de menor
    numeración ('ECS 10'). En colisión de hash la entrada posterior sobreescribe
    la anterior, es decir gana el re-rateo más reciente del chart.

    Pasos de post-procesamiento aplicados antes de devolver:
    - Hash MD5 truncado a 8 hex chars y guardado como `chart_id` (primera columna)
    - Columnas reordenadas siguiendo la convención de la hoja de cálculo comunitaria.

    Args:
        raw_dir: Directorio raíz que contiene los packs/charts .sm.
                 / Root directory containing .sm pack/chart folders.

    Returns:
        DataFrame con una fila por chart único (deduplicado por hash de notas).
        / DataFrame with one row per unique chart (deduplicated by note hash).

    Raises:
        FileNotFoundError: Si raw_dir no existe. / If raw_dir does not exist.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    # ES: Ordenar por ruta para garantizar reproducibilidad y orden de overwrite.
    # EN: Sort by path string for reproducibility and consistent overwrite order.
    sm_files: List[Path] = sorted(raw_dir.rglob("*.sm"), key=lambda p: str(p))
    total_files = len(sm_files)
    logger.info("Found %d .sm files in '%s'", total_files, raw_dir)

    # ES: Dict con hash MD5 como clave; re-rateos sobreescriben entradas antiguas.
    # EN: Dict keyed by MD5 hash; re-rates overwrite older entries.
    chart_by_hash: Dict[str, Dict] = {}
    duplicates_overwritten = 0

    for file_path in tqdm(sm_files, desc="Processing charts", unit="chart"):
        record = process_file(file_path, raw_dir)
        if record is None:
            continue

        notes_hash: str = record["_notes_hash"]
        if notes_hash in chart_by_hash:
            duplicates_overwritten += 1

        chart_by_hash[notes_hash] = record

    logger.info(
        "Parsed %d/%d files — %d unique charts (%d duplicates overwritten)",
        sum(1 for r in chart_by_hash.values()),
        total_files,
        len(chart_by_hash),
        duplicates_overwritten,
    )

    if not chart_by_hash:
        logger.warning("No charts were successfully parsed. DataFrame will be empty.")
        return pd.DataFrame()

    df = pd.DataFrame(list(chart_by_hash.values()))

    # ES: Convertir el hash MD5 completo en un identificador corto de 8 caracteres
    #     y renombrarlo a 'chart_id'. 
    #     No es una característica de ML — se coloca al inicio para trazabilidad.
    # EN: Shorten the full MD5 to an 8-character identifier and rename it to
    #     'chart_id'.
    #     Not an ML feature — placed first for traceability.
    df["chart_id"] = df["_notes_hash"].str[:8]
    df.drop(columns=["_notes_hash"], inplace=True)

    # ES: Filtro de control de calidad: descartar charts con stream total menor al
    #     umbral mínimo. Esto elimina registros donde la subdivisión fue mal detectada
    #     (p.ej. charts con bursts o notas especiales que confunden al parser).
    #     Este paso es metodológicamente justificable en la tesis como QA de datos.
    # EN: Quality control filter: discard charts whose total stream length is below
    #     the minimum threshold. This removes records where subdivision was
    #     misdetected (e.g. charts with bursts or special notes that confuse the
    #     parser). This step is methodologically justifiable as data QA.
    before_filter = len(df)
    df = df[df["total_stream_length"] > MIN_STREAM_LENGTH].reset_index(drop=True)
    filtered_out = before_filter - len(df)
    if filtered_out > 0:
        logger.warning(
            "Quality filter removed %d charts with total_stream_length < %d "
            "(likely subdivision misdetection). %d charts remain.",
            filtered_out,
            MIN_STREAM_LENGTH,
            len(df),
        )

    # ES: Reordenar columnas siguiendo la convención de la hoja de cálculo
    #     comunitaria: identificación → velocidad → volumen de stream → ratio → densidad.
    # EN: Reorder columns following the community spreadsheet convention:
    #     identification → speed → stream volume → ratio → density.
    ordered_cols = [
        "chart_id",
        "difficulty",
        "breakdown",
        "source_file",
        "display_bpm",
        "ebpm",
        "total_stream_length",
        "max_stream_length",
        "break_count",
        "stream_break_ratio",
        "average_nps",
    ]
    # ES: Incluir columnas extra no listadas explícitamente al final (futuro-compatible).
    # EN: Append any unlisted extra columns at the end (future-compatible).
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra]

    return df


def parse_args() -> argparse.Namespace:
    """
    English: Parse CLI arguments for build_dataset automation.
    Español: Analiza los argumentos de línea de comandos para la automatización
    de build_dataset.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Parse all .sm stamina charts under ml-core/data/raw/ and save a "
            "training CSV to ml-core/data/processed/stamina_dataset.csv."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Root directory containing .sm packs (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Destination CSV path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """
    English: Entry point — build the dataset and save to CSV.
    Español: Punto de entrada — construye el dataset y lo guarda en CSV.
    """

    args = parse_args()
    raw_dir: Path = args.raw_dir
    output_path: Path = args.output

    logger.info("Starting dataset build from '%s'", raw_dir)

    df = build_dataset(raw_dir)

    if df.empty:
        logger.error("Dataset is empty; nothing saved.")
        sys.exit(1)

    # ES: Crear el directorio de salida si no existe.
    # EN: Create the output directory if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Dataset saved → '%s'  (%d rows x %d columns)",
        output_path,
        len(df),
        len(df.columns),
    )


if __name__ == "__main__":
    main()
