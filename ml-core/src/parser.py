from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

_NOTE_SYMBOLS = {"0", "1", "2", "3", "4", "M"}
_NOTE_HEADS = {"1", "2", "4"}
_DIFFICULTY_PRIORITY = {"challenge": 2, "hard": 1}


def _strip_inline_comment(line: str) -> str:
    """ES: Elimina comentarios en línea que comienzan con //.

    EN: Remove inline comments that start with //.
    """

    return line.split("//", 1)[0].rstrip()


def _load_clean_lines(file_path: Path) -> List[str]:
    """ES: Carga un archivo .sm y elimina las líneas de comentario que comienzan con //.

    EN: Load a .sm file, dropping comment lines that start with //.

    Args:
        file_path: Path to the .sm chart file.

    Returns:
        List of lines with comment lines removed and line endings stripped.
    """

    text = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if line.lstrip().startswith("//"):
            continue
        cleaned.append(line)
    return cleaned


def _extract_tag_payload(lines: Sequence[str], tag: str) -> Optional[str]:
    """ES: Extrae el contenido de un tag #TAG:...; manejando líneas múltiples.

    EN: Extract the payload of a #TAG:...; entry, handling multiline values.
    """

    tag_prefix = f"{tag.upper()}:"
    for idx, raw in enumerate(lines):
        line = _strip_inline_comment(raw).strip()
        if not line:
            continue
        if not line.upper().startswith(tag_prefix):
            continue

        payload = line.split(":", 1)[1]
        if ";" in payload:
            return payload.split(";", 1)[0].strip()

        parts = [payload]
        cursor = idx + 1
        while cursor < len(lines):
            cont = _strip_inline_comment(lines[cursor]).strip()
            if not cont:
                cursor += 1
                continue
            parts.append(cont)
            if ";" in cont:
                joined = " ".join(parts)
                return joined.split(";", 1)[0].strip()
            cursor += 1

        return " ".join(parts).strip()

    return None


def _parse_display_bpm(payload: Optional[str]) -> Optional[float]:
    """ES: Convierte el payload de #DISPLAYBPM a un float si es posible.

    EN: Convert a #DISPLAYBPM payload to float when possible.
    """

    if not payload:
        return None

    normalized = payload.strip()
    if normalized in {"*", "0", "0.0", "0.000"}:
        return None

    match = re.search(r"[-+]?\d*\.?\d+", normalized)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_bpms(payload: Optional[str]) -> List[float]:
    """ES: Extrae valores BPM desde un payload #BPMS (beat=bpm,...).

    EN: Extract BPM values from a #BPMS payload (beat=bpm,...).
    """

    if not payload:
        return []

    bpms: List[float] = []
    for part in payload.split(","):
        entry = part.strip()
        if not entry:
            continue
        if "=" not in entry:
            continue
        _, bpm_text = entry.split("=", 1)
        try:
            bpms.append(float(bpm_text))
        except ValueError:
            continue
    return bpms


def _select_display_bpm(lines: Sequence[str]) -> Optional[float]:
    """ES: Determina el Display BPM usando #DISPLAYBPM o la moda de #BPMS.

    EN: Determine Display BPM using #DISPLAYBPM or the mode of #BPMS entries.
    """

    display_payload = _extract_tag_payload(lines, "#DISPLAYBPM")
    display_bpm = _parse_display_bpm(display_payload)
    if display_bpm is not None:
        return display_bpm

    bpm_payload = _extract_tag_payload(lines, "#BPMS")
    bpms = _parse_bpms(bpm_payload)
    if not bpms:
        return None

    rounded = [round(bpm, 3) for bpm in bpms]
    most_common = Counter(rounded).most_common(1)[0][0]
    return float(most_common)


def _extract_charts(lines: Sequence[str]) -> List[dict]:
    """ES: Extrae metadatos del chart y bloques de notas desde líneas de un .sm.

    EN: Extract chart metadata and note data blocks from a list of .sm lines.
    """

    charts: List[dict] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("#NOTES"):
            i += 1
            continue

        if i + 5 >= len(lines):
            break

        chart_type = lines[i + 1].strip().rstrip(":").lower()
        description = lines[i + 2].strip().rstrip(":")
        difficulty = lines[i + 3].strip().rstrip(":").lower()
        meter_text = lines[i + 4].strip().rstrip(":")
        radar = lines[i + 5].strip().rstrip(":")

        try:
            meter = int(meter_text)
        except ValueError:
            meter = 0

        note_lines: List[str] = []
        j = i + 6
        while j < len(lines):
            current = _strip_inline_comment(lines[j].strip())
            if not current:
                j += 1
                continue
            if current.endswith(";"):
                tail = current[:-1].strip()
                if tail:
                    note_lines.append(tail)
                j += 1
                break
            note_lines.append(current)
            j += 1

        charts.append(
            {
                "type": chart_type,
                "description": description,
                "difficulty": difficulty,
                "meter": meter,
                "radar": radar,
                "note_lines": note_lines,
            }
        )

        i = j

    return charts


def _select_hardest_chart(charts: Sequence[dict]) -> dict:
    """ES: Selecciona el chart más difícil tipo dance-single (Challenge sobre Hard).

    EN: Pick the hardest dance-single chart (Challenge preferred over Hard).
    """

    selected: dict | None = None
    selected_priority = -1
    selected_meter = -1

    for chart in charts:
        if chart.get("type") != "dance-single":
            continue
        difficulty = chart.get("difficulty", "")
        priority = _DIFFICULTY_PRIORITY.get(difficulty)
        if priority is None:
            continue

        meter = chart.get("meter", 0)
        if (
            selected is None
            or priority > selected_priority
            or (priority == selected_priority and meter > selected_meter)
        ):
            selected = chart
            selected_priority = priority
            selected_meter = meter

    if selected is None:
        raise ValueError("No suitable dance-single chart with difficulty Hard or Challenge was found.")

    return selected


def _count_notes_per_measure(note_lines: Sequence[str]) -> List[int]:
    """ES: Convierte las líneas de notas en un conteo de notas por compás.

    EN: Convert raw note lines into a per-measure note count list.
    """

    measures = _prepare_measures(note_lines)
    densities, _ = _count_notes_and_rows(measures)
    return densities


def _prepare_measures(note_lines: Sequence[str]) -> List[List[str]]:
    """ES: Estandariza los bloques de medidas limpiando comentarios y vacíos.

    EN: Normalize measure blocks by stripping inline comments and empty rows.
    """

    measures: List[List[str]] = []
    for block in "\n".join(note_lines).split(","):
        rows: List[str] = []
        for raw in block.splitlines():
            row = _strip_inline_comment(raw).strip()
            if row:
                rows.append(row)
        measures.append(rows)
    return measures


def _count_notes_and_rows(measures: Sequence[Sequence[str]]) -> Tuple[List[int], List[int]]:
    """ES: Cuenta notas y filas por medida.

    EN: Count notes and row lengths per measure.
    """

    densities: List[int] = []
    row_counts: List[int] = []

    for measure_index, rows in enumerate(measures):
        row_counts.append(len(rows))
        measure_count = 0
        for row in rows:
            if len(row) != 4:
                raise ValueError(
                    f"Invalid row length in measure {measure_index}: '{row}' (expected 4 columns)."
                )
            invalid_symbols = set(row) - _NOTE_SYMBOLS
            if invalid_symbols:
                raise ValueError(
                    f"Unsupported symbols {invalid_symbols} in measure {measure_index}: '{row}'."
                )
            measure_count += sum(1 for ch in row if ch in _NOTE_HEADS)
        densities.append(measure_count)

    return densities, row_counts


def _infer_subdivision(row_counts: Sequence[int]) -> int:
    """ES: Infere la subdivisión base (16ths, 24ths, etc.) usando la moda de filas.

    EN: Infer base subdivision (e.g., 16ths, 24ths) using the mode of row counts.
    """

    non_zero = [c for c in row_counts if c > 0]
    if not non_zero:
        return 16
    return Counter(non_zero).most_common(1)[0][0]


def parse_sm_chart_with_meta(file_path: Path) -> Tuple[Dict[str, Optional[float] | List[int]], int]:
    """ES: Parsea un .sm y devuelve notas por compás, Display BPM y subdivisión.

    EN: Parse an .sm and return per-measure notes, Display BPM, and subdivision.
    """

    lines = _load_clean_lines(file_path)
    display_bpm = _select_display_bpm(lines)
    charts = _extract_charts(lines)
    chart = _select_hardest_chart(charts)

    measures = _prepare_measures(chart["note_lines"])
    densities, row_counts = _count_notes_and_rows(measures)
    subdivision = _infer_subdivision(row_counts)
    notes_data = {"display_bpm": display_bpm, "notes_per_measure": densities}
    return notes_data, subdivision


def parse_sm_chart(file_path: Path) -> Dict[str, Optional[float] | List[int]]:
    """ES: Parsea un archivo .sm de StepMania y produce notas por compás y Display BPM.

    El parser elimina comentarios, extrae los bloques `#NOTES`, selecciona el chart
    dance-single más difícil (Challenge sobre Hard) y cuenta las cabezas de nota
    (`1`, `2`, `4`) por compás.

    EN: Parse a StepMania .sm file into per-measure note density counts. The parser
    removes comment lines, extracts all `#NOTES` blocks, selects the hardest
    available `dance-single` chart (Challenge preferred over Hard), and counts
    note heads (`1`, `2`, `4`) per measure.

    Args:
        file_path: Path to a StepMania .sm file.

    Returns:
        Dict con `display_bpm` y `notes_per_measure` para el chart seleccionado.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no suitable chart is found or the note data is malformed.
    """

    notes_data, _ = parse_sm_chart_with_meta(file_path)
    return notes_data


if __name__ == "__main__":
    test_path = Path("ml-core/data/raw/Stamina RPG 6/[26] Stratospheric Intricacy/stratospheric.sm")
    try:
        notes_data = parse_sm_chart(test_path)
    except FileNotFoundError:
        print(f"File not found: {test_path}")
    except ValueError as exc:
        print(f"Failed to parse chart: {exc}")
    else:
        print("Per-measure note density:")
        print(notes_data)
