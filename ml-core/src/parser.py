from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

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


def parse_sm_chart_with_meta(file_path: Path) -> Tuple[List[int], int]:
    """ES: Parsea un .sm y devuelve densidades y la subdivisión predominante.

    EN: Parse an .sm and return densities plus the predominant subdivision.
    """

    lines = _load_clean_lines(file_path)
    charts = _extract_charts(lines)
    chart = _select_hardest_chart(charts)

    measures = _prepare_measures(chart["note_lines"])
    densities, row_counts = _count_notes_and_rows(measures)
    subdivision = _infer_subdivision(row_counts)
    return densities, subdivision


def parse_sm_chart(file_path: Path) -> List[int]:
    """ES: Parsea un archivo .sm de StepMania y produce el conteo de notas por compás.

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
        List of note counts per measure for the selected chart.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no suitable chart is found or the note data is malformed.
    """

    densities, _ = parse_sm_chart_with_meta(file_path)
    return densities


if __name__ == "__main__":
    test_path = Path("ml-core/data/raw/Stamina RPG 6/[26] Stratospheric Intricacy/stratospheric.sm")
    try:
        densities = parse_sm_chart(test_path)
    except FileNotFoundError:
        print(f"File not found: {test_path}")
    except ValueError as exc:
        print(f"Failed to parse chart: {exc}")
    else:
        print("Per-measure note density:")
        print(densities)
