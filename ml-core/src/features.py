from __future__ import annotations

from pathlib import Path
from pprint import pprint
from statistics import mean
from typing import Dict, List

StreamSegment = Dict[str, int | bool]


def get_stream_sequences(notes_per_measure: List[int], threshold: int = 16) -> List[StreamSegment]:
    """ES: Identifica rangos contiguos de stream y break según un umbral de notas.

    Replica el helper `GetStreamSequences` de Simply Love con índices 0-based.
    Un compás es *stream* si su conteo de notas es >= `threshold`. Los compases
    contiguos forman un segmento de stream. Las pausas entre streams se registran
    solo si duran al menos 2 compases, igual que la lógica en Lua.

    EN: Identify contiguous stream and break ranges based on a note threshold.
    Mirrors Simply Love’s `GetStreamSequences` with Pythonic 0-based indices. A
    measure is a stream if its note count is >= `threshold`. Consecutive stream
    measures form a stream segment. Breaks between streams are recorded when they
    are at least 2 measures long, matching the Lua logic that ignores single-measure
    breaks.

    Args:
        notes_per_measure: Note counts per measure from the parser.
        threshold: Minimum notes-per-measure to treat as a stream measure.

    Returns:
        List of segments, each as a dict with keys: `start` (inclusive), `end`
        (inclusive), `is_break`, and `length` (number of measures in the segment).
    """

    stream_measures = [i for i, n in enumerate(notes_per_measure) if n >= threshold]
    if not stream_measures:
        return []

    stream_sequences: List[StreamSegment] = []
    stream_sequence_threshold = 1
    break_sequence_threshold = 2

    # Leading break before the first stream segment, if it is long enough.
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
            stream_end = cur_val + 1  # tentative end (inclusive) as we extend the run
            continue

        # Finalize current stream run.
        if stream_end is None:
            stream_end = cur_val
        stream_start = stream_end - counter + 1

        if counter >= stream_sequence_threshold:
            length = stream_end - stream_start + 1
            stream_sequences.append(
                {"start": stream_start, "end": stream_end, "is_break": False, "length": length}
            )

        # Add an intermediate or trailing break if it is long enough.
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

    return stream_sequences


def calculate_breakdown_metrics(notes_per_measure: List[int]) -> Dict[str, float | int]:
    """ES: Calcula métricas resumen a partir de la densidad de notas por compás.

    EN: Compute summary metrics from per-measure note densities, mirroring the Lua
    helper for downstream ML features.

    Args:
        notes_per_measure: Note counts per measure from the parser.

    Returns:
        Diccionario/Dictionary con estadísticas agregadas: `total_stream_length`,
        `max_stream_length`, `break_count`, `stream_break_ratio`, y `average_nps`.
    """

    sequences = get_stream_sequences(notes_per_measure, threshold=16)

    total_stream_length = sum(seg["length"] for seg in sequences if not seg["is_break"])
    total_break_length = sum(seg["length"] for seg in sequences if seg["is_break"])
    max_stream_length = max((seg["length"] for seg in sequences if not seg["is_break"]), default=0)
    break_count = sum(1 for seg in sequences if seg["is_break"])

    stream_break_ratio = (
        float(total_stream_length) / float(total_break_length) if total_break_length > 0 else float("inf")
    )

    average_nps = mean(notes_per_measure) if notes_per_measure else 0.0

    return {
        "total_stream_length": total_stream_length,
        "max_stream_length": max_stream_length,
        "break_count": break_count,
        "stream_break_ratio": stream_break_ratio,
        "average_nps": average_nps,
    }


def generate_breakdown_string(notes_per_measure: List[int], threshold: int = 16) -> str:
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

    segments = get_stream_sequences(notes_per_measure, threshold=threshold)
    if not segments:
        return "No streams detected"

    parts: List[str] = []
    for seg in segments:
        if seg["is_break"]:
            parts.append(f"({seg['length']})")
        else:
            parts.append(str(seg["length"]))
    return " ".join(parts)


if __name__ == "__main__":
    dummy_path = Path("ml-core/data/raw/Stamina RPG 6/[19] lovism/lovism.sm")
    try:
        from parser import parse_sm_chart

        densities = parse_sm_chart(dummy_path)
        print("Stream sequences:")
        seqs = get_stream_sequences(densities)
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
        pprint(calculate_breakdown_metrics(densities))
        print("\nBreakdown string:")
        print(generate_breakdown_string(densities))
    except Exception as exc:  # noqa: BLE001 - CLI helper only
        print(f"Failed to compute features: {exc}")
