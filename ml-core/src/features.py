from __future__ import annotations

from pathlib import Path
from pprint import pprint
from statistics import mean
from typing import Dict, List, Optional, Union

STREAM_THRESHOLD = 16

StreamSegment = Dict[str, int | float | bool]

"""
-------------------------------------------------------------------------------
PORTED LOGIC DISCLAIMER - DISCLAIMER DE LÓGICA PORTEADA
-------------------------------------------------------------------------------

[EN]
The logic for stream sequence detection in this module is a direct Python port 
of the Lua code from the 'Simply Love' StepMania Theme.
This adaptation is distributed under the GNU General Public License v3.0 (GPLv3)
to comply with the original license terms.

Original Source: Simply-Love-SM5 (https://github.com/zarzob/Simply-Love-SM5)
Original File:   Scripts/SL-ChartParserHelpers.lua
Copyright (C):   2020 Simply Love Team
License:         GNU GPLv3

Modifications:
- Translated from Lua to Python 3.10+ by José Carlos Macías Macías for academic thesis purposes.
- Adjusted array indexing (1-based to 0-based).
- Integrated into a Scikit-Learn pipeline context.

-------------------------------------------------------------------------------

[ES]
La lógica para la detección de secuencias de 'stream' en este módulo es un porte
directo a Python del código Lua del tema 'Simply Love' de StepMania.
Esta adaptación se distribuye bajo la Licencia Pública General de GNU v3.0 (GPLv3)
para cumplir con los términos de la licencia original.

Fuente Original: Simply-Love-SM5 (https://github.com/Simply-Love/Simply-Love-SM5)
Archivo Orig.:   Scripts/SL-ChartParserHelpers.lua
Copyright (C):   2020 Simply Love Team
Licencia:        GNU GPLv3

Modificaciones:
- Traducido de Lua a Python 3.10+ por José Carlos Macías Macías para fines de tesis académica.
- Ajuste de índices de arrays (base-1 a base-0).
- Integración en un contexto de pipeline de Scikit-Learn.
-------------------------------------------------------------------------------
"""
def _build_segments(notes_per_measure: List[int], threshold: int) -> List[StreamSegment]:
    """ES: Construye segmentos stream/break con un umbral dado (helper interno).

    Quita los breaks iniciales/finales para imitar el comportamiento del tema
    StepMania.

    EN: Build raw stream/break segments for a given threshold (internal helper).
    Removes leading/trailing breaks to mirror the StepMania theme behavior.
    """

    stream_measures = [i for i, n in enumerate(notes_per_measure) if n >= threshold]
    if not stream_measures:
        return []

    stream_sequences: List[StreamSegment] = []
    stream_sequence_threshold = 1
    break_sequence_threshold = 2

    break_start = 0
    first_stream = stream_measures[0]
    break_end = first_stream - 1
    break_len = break_end - break_start + 1
    if break_len >= break_sequence_threshold:
        stream_sequences.append(
            {"start": break_start, "end": break_end, "is_break": True, "length": break_len}
        )

    counter = 1
    stream_end: int | None = None

    for idx, cur_val in enumerate(stream_measures):
        next_val = stream_measures[idx + 1] if idx + 1 < len(stream_measures) else None

        if next_val is not None and cur_val + 1 == next_val:
            counter += 1
            stream_end = cur_val + 1
            continue

        if stream_end is None:
            stream_end = cur_val
        stream_start = stream_end - counter + 1

        if counter >= stream_sequence_threshold:
            length = stream_end - stream_start + 1
            stream_sequences.append(
                {"start": stream_start, "end": stream_end, "is_break": False, "length": length}
            )

        if next_val is not None:
            break_start = cur_val + 1
            break_end = next_val - 1
        else:
            break_start = cur_val + 1
            break_end = len(notes_per_measure) - 1

        break_len = break_end - break_start + 1
        if break_len >= break_sequence_threshold:
            stream_sequences.append(
                {"start": break_start, "end": break_end, "is_break": True, "length": break_len}
            )

        counter = 1
        stream_end = None

    if stream_sequences and stream_sequences[0]["is_break"]:
        stream_sequences = stream_sequences[1:]
    if stream_sequences and stream_sequences[-1]["is_break"]:
        stream_sequences = stream_sequences[:-1]

    return stream_sequences


def _compute_density(segments: List[StreamSegment], multiplier: float) -> float:
    """ES: Calcula la densidad de stream escalada igual que el helper de Lua.

    EN: Compute scaled stream density mirroring the Simply Love Lua helper.
    """

    scaled_stream = sum(seg["length"] * multiplier for seg in segments if not seg["is_break"])
    scaled_total = sum(seg["length"] * multiplier for seg in segments)
    if scaled_total == 0:
        return 0.0
    return scaled_stream / scaled_total


def get_stream_sequences(
    notes_per_measure: List[int], subdivision: Optional[int] = None
) -> List[StreamSegment]:
    """ES: Detecta secuencias de stream/break usando la lógica del tema Simply Love.

    Implementa el mismo flujo que `GenerateBreakdownText` del Lua original: se
    prueban umbrales altos (32, 24, 20) para captar charts en 24ths/32nds y se
    cae a 16 como último recurso. Cada candidato se acepta solo si la densidad
    de stream escalada es >= 0.2. Los largos se escalan con un multiplicador
    equivalente a `threshold/16` para reportar longitudes comparables a 16ths.

    EN: Detect stream/break segments following Simply Love's display logic. It
    tries higher thresholds (32, 24, 20) to accommodate 24th/32nd charts and
    falls back to 16. A candidate is accepted only if the scaled stream density
    is >= 0.2. Lengths are scaled by `threshold/16` to keep them comparable to
    16th-based breakdowns.

    Args:
        notes_per_measure: Note counts per measure from the parser.
        subdivision: Rows-per-measure grid (metadata only; no direct scaling).

    Returns:
        Lista/List of segment dicts with keys: `start`, `end`, `is_break`,
        `length` (raw measures), `scaled_length` (equivalent 16th-length), and
        the selected `threshold`.
    """

    candidate_configs = [
        {"threshold": 32, "multiplier": 2.0},
        {"threshold": 24, "multiplier": 1.5},
        {"threshold": 20, "multiplier": 1.25},
        {"threshold": STREAM_THRESHOLD, "multiplier": 1.0},
    ]

    chosen_segments: List[StreamSegment] = []
    chosen_multiplier = 1.0
    chosen_threshold = STREAM_THRESHOLD

    for config in candidate_configs:
        segments = _build_segments(notes_per_measure, threshold=config["threshold"])
        if not segments:
            continue

        density = _compute_density(segments, multiplier=config["multiplier"])
        if density < 0.2:
            continue

        chosen_segments = segments
        chosen_multiplier = config["multiplier"]
        chosen_threshold = config["threshold"]
        break

    if not chosen_segments:
        chosen_segments = _build_segments(notes_per_measure, threshold=STREAM_THRESHOLD)
        chosen_multiplier = 1.0
        chosen_threshold = STREAM_THRESHOLD

    for seg in chosen_segments:
        scaled_len = int(seg["length"] * chosen_multiplier)
        seg.update(
            {
                "scaled_length": max(1, scaled_len),
                "threshold": chosen_threshold,
                "multiplier": chosen_multiplier,
            }
        )

    return chosen_segments


def calculate_ebpm_profile(
    notes_data: Dict[str, Optional[float] | List[int]]
) -> Dict[str, Optional[float] | List[Optional[float]]]:
    """ES: Calcula el perfil de NPS y eBPM por compás a partir de Display BPM.

    Si no hay Display BPM válido, devuelve listas de None para evitar divisiones
    por cero.

    EN: Compute per-measure NPS and eBPM profile from a Display BPM. If Display BPM
    is missing/invalid, return None-filled lists to avoid division by zero.

    Args:
        notes_data: Dict con `display_bpm` y `notes_per_measure`.

    Returns:
        Dict con `display_bpm`, `measure_seconds`, `nps_per_measure`, y
        `ebpm_per_measure`.
    """

    notes_per_measure = notes_data.get("notes_per_measure", [])
    display_bpm = notes_data.get("display_bpm")

    if not display_bpm or display_bpm <= 0:
        return {
            "display_bpm": display_bpm,
            "measure_seconds": None,
            "nps_per_measure": [None for _ in notes_per_measure],
            "ebpm_per_measure": [None for _ in notes_per_measure],
        }

    measure_seconds = 240.0 / display_bpm
    nps_per_measure = [count / measure_seconds for count in notes_per_measure]
    ebpm_per_measure = [nps * 15.0 for nps in nps_per_measure]

    return {
        "display_bpm": display_bpm,
        "measure_seconds": measure_seconds,
        "nps_per_measure": nps_per_measure,
        "ebpm_per_measure": ebpm_per_measure,
    }


def calculate_breakdown_metrics(
    notes_input: Union[List[int], Dict[str, Optional[float] | List[int]]],
    subdivision: Optional[int] = None,
) -> Dict[str, float | int | None]:
    """ES: Calcula métricas resumen a partir de la densidad de notas por compás.

    EN: Compute summary metrics from per-measure note densities, mirroring the Lua
    helper for downstream ML features.

    Args:
        notes_input: Lista de notas por compás o dict con `display_bpm` y
            `notes_per_measure`.

    Returns:
        Diccionario/Dictionary con estadísticas agregadas: `total_stream_length`,
        `max_stream_length`, `break_count`, `stream_break_ratio`, `average_nps`,
        `peak_ebpm`, y `average_ebpm`.
    """

    if isinstance(notes_input, dict):
        notes_per_measure = notes_input.get("notes_per_measure", [])
        display_bpm = notes_input.get("display_bpm")
    else:
        notes_per_measure = notes_input
        display_bpm = None

    sequences = get_stream_sequences(notes_per_measure, subdivision=subdivision)

    total_stream_length = sum(seg.get("scaled_length", seg["length"]) for seg in sequences if not seg["is_break"])
    total_break_length = sum(seg.get("scaled_length", seg["length"]) for seg in sequences if seg["is_break"])
    max_stream_length = max((seg.get("scaled_length", seg["length"]) for seg in sequences if not seg["is_break"]), default=0)
    break_count = sum(1 for seg in sequences if seg["is_break"])

    stream_break_ratio = (
        float(total_stream_length) / float(total_break_length) if total_break_length > 0 else float("inf")
    )

    if display_bpm and display_bpm > 0:
        measure_seconds = 240.0 / display_bpm
        average_nps = mean(
            [count / measure_seconds for count in notes_per_measure]
        ) if notes_per_measure else 0.0
    else:
        average_nps = mean(notes_per_measure) if notes_per_measure else 0.0

    peak_ebpm: Optional[float] = None
    average_ebpm: Optional[float] = None
    if display_bpm and display_bpm > 0 and notes_per_measure:
        measure_seconds = 240.0 / display_bpm
        ebpm_per_measure = [
            (count / measure_seconds) * 15.0 for count in notes_per_measure
        ]
        stream_indices: List[int] = []
        for seg in sequences:
            if seg["is_break"]:
                continue
            start = max(0, int(seg["start"]))
            end = min(len(notes_per_measure) - 1, int(seg["end"]))
            if start <= end:
                stream_indices.extend(range(start, end + 1))

        if stream_indices:
            stream_ebpm = [ebpm_per_measure[i] for i in stream_indices]
            peak_ebpm = max(stream_ebpm)
            average_ebpm = mean(stream_ebpm)

    return {
        "total_stream_length": total_stream_length,
        "max_stream_length": max_stream_length,
        "break_count": break_count,
        "stream_break_ratio": stream_break_ratio,
        "average_nps": average_nps,
        "peak_ebpm": peak_ebpm,
        "average_ebpm": average_ebpm,
    }


def generate_breakdown_string(
    notes_per_measure: List[int], subdivision: Optional[int] = None
) -> str:
    """ES: Genera una cadena compacta de breakdown usando segmentos stream/break.

    Los streams se muestran como su longitud en compases y los breaks entre
    paréntesis. Es una versión simplificada del texto de breakdown de Simply Love
    para visualización rápida en la tesis.

    EN: Create a compact breakdown string using stream/break segments. Streams are
    shown by length in measures; breaks use parentheses. Simplified Simply Love
    breakdown text for quick visualization/evidence.

    Example output: "20 (2) 30 (8) 16"

    Args:
        notes_per_measure: Note counts per measure from the parser.
        threshold: Minimum notes-per-measure to treat as a stream measure.

    Returns:
        A single-line human-readable breakdown string.
    """

    segments = get_stream_sequences(notes_per_measure, subdivision=subdivision)
    if not segments:
        return "No streams detected"

    parts: List[str] = []
    for seg in segments:
        display_len = seg.get("scaled_length", seg["length"])
        if seg["is_break"]:
            parts.append(f"({display_len})")
        else:
            parts.append(str(display_len))
    return " ".join(parts)


if __name__ == "__main__":
    dummy_path = Path("ml-core/data/raw/Stamina RPG 6/[23] Cycle Hit/Cycle Hit.sm")
    try:
        from parser import parse_sm_chart_with_meta

        notes_data, subdivision = parse_sm_chart_with_meta(dummy_path)
        densities = notes_data["notes_per_measure"]
        print("Stream sequences:")
        seqs = get_stream_sequences(densities, subdivision=subdivision)
        formatted_sequences = [
            {
                "length": seg["length"],
                "is_break": seg["is_break"],
                "start": seg["start"],
                "end": seg["end"],
            }
            for seg in seqs
        ]
        pprint(formatted_sequences, sort_dicts=False)
        print("\nMetrics:")
        pprint(calculate_breakdown_metrics(notes_data, subdivision=subdivision))
        print("\nBreakdown string:")
        print(generate_breakdown_string(densities, subdivision=subdivision))
    except Exception as exc:  # noqa: BLE001 - CLI helper only
        print(f"Failed to compute features: {exc}")
