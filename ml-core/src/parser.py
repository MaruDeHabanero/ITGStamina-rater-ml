from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple
from typing_extensions import TypedDict


class NotesData(TypedDict):
    """ES: Resultado del parseo de un .sm.

    Campos:
        display_bpm:  BPM extraído de #TITLE ([dificultad] [bpm] nombre).
                      Es la velocidad de referencia que la comunidad utiliza
                      para describir el chart.
        timing_bpm:   BPM extraído de #BPMS (moda). Es el BPM base real del
                      archivo, usado únicamente para calcular duración de
                      compases y NPS. Puede diferir de display_bpm en charts
                      de 24ths o 32nds donde el engine corre a un BPM base
                      distinto al eBPM percibido.
        block:        Dificultad numérica del chart (variable objetivo).
        notes_per_measure: Notas por compás del chart seleccionado.

    EN: Parsed .sm result.

    Fields:
        display_bpm:  BPM from #TITLE ([difficulty] [bpm] name).
                      The community-facing speed reference for the chart.
        timing_bpm:   BPM from #BPMS (mode). The actual base BPM of the file,
                      used only to compute measure durations and NPS. May differ
                      from display_bpm for 24th/32nd charts where the engine
                      runs at a different base BPM than the perceived eBPM.
        block:        Numeric difficulty label (classification target).
        notes_per_measure: Per-measure note counts for the selected chart.
    """

    display_bpm: Optional[float]
    timing_bpm: Optional[float]
    block: int
    notes_per_measure: List[int]

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


# ES: Subdivisiones válidas en charts de stamina ITG.
#     16 = 16ths (estándar), 20 = 20ths (rara), 24 = 24ths (popular),
#     32 = 32nds (muy rara, equivale a 2× el BPM).
# EN: Valid subdivisions in ITG stamina charts.
#     16 = 16ths (standard), 20 = 20ths (rare), 24 = 24ths (popular),
#     32 = 32nds (very rare, effectively 2× the BPM).
_VALID_SUBDIVISIONS = [16, 20, 24, 32]

# ES: Regex para extraer BPM del #TITLE: [dificultad] [bpm] nombre
#     Ejemplo: "[42] [465] Nocturnal 2097" → grupo 1 = "465"
# EN: Regex to extract BPM from #TITLE: [difficulty] [bpm] name
#     Example: "[42] [465] Nocturnal 2097" → group 1 = "465"
_TITLE_BPM_RE = re.compile(r"\[\d+\]\s*\[(\d+(?:\.\d+)?)\]")


def _parse_bpms(payload: Optional[str]) -> List[float]:
    """ES: Extrae valores BPM desde un payload #BPMS (beat=bpm,...).

    EN: Extract BPM values from a #BPMS payload (beat=bpm,...).
    """

    if not payload:
        return []

    bpms: List[float] = []
    for part in payload.split(","):
        entry = part.strip()
        if not entry or "=" not in entry:
            continue
        _, bpm_text = entry.split("=", 1)
        try:
            bpms.append(float(bpm_text.strip()))
        except ValueError:
            continue
    return bpms


def _extract_timing_bpm(lines: Sequence[str]) -> Optional[float]:
    """ES: Extrae el BPM de timing (base) desde el tag #BPMS usando su moda.

    Este BPM es el que el engine de StepMania usa para medir la duración real
    de los compases. Se usa únicamente para cálculos internos de NPS/eBPM,
    y nunca se expone como velocidad del chart al usuario.

    En charts con subdivisión no estándar (ej. 32nds) el timing_bpm será
    distinto al display_bpm (ej. Flaklypa: timing=130, display=260).

    EN: Extract the base timing BPM from #BPMS using its mode.

    This BPM is what the StepMania engine uses to measure actual measure
    durations. Used only for internal NPS/eBPM calculations and never
    exposed as the chart speed to the user.

    For non-standard subdivision charts (e.g. 32nds) timing_bpm will differ
    from display_bpm (e.g. Flaklypa: timing=130, display=260).

    Args:
        lines: Líneas limpias del archivo .sm. / Clean lines from the .sm file.

    Returns:
        Moda de los BPMs en #BPMS como float, o None si no hay datos.
        / Mode of #BPMS entries as float, or None if unavailable.
    """

    bpm_payload = _extract_tag_payload(lines, "#BPMS")
    bpms = _parse_bpms(bpm_payload)
    if not bpms:
        return None
    rounded = [round(b, 3) for b in bpms]
    return float(Counter(rounded).most_common(1)[0][0])


def _extract_title_bpm(lines: Sequence[str]) -> Optional[float]:
    """ES: Extrae el BPM del tag #TITLE con formato [dificultad] [bpm] nombre.

    En packs de stamina ITG, el #TITLE codifica tanto la dificultad como el BPM
    de referencia del chart (el eBPM percibido por la comunidad). Este valor
    es la fuente autoritativa de velocidad y puede diferir del timing_bpm
    (#BPMS) en charts con subdivisiones no estándar (24ths, 32nds).

    EN: Extract BPM from the #TITLE tag with format [difficulty] [bpm] name.

    In ITG stamina packs, #TITLE encodes both the difficulty and the reference
    BPM of the chart (the community-perceived eBPM). This value is the
    authoritative speed reference and may differ from timing_bpm (#BPMS) for
    charts with non-standard subdivisions (24ths, 32nds).

    Example:
        #TITLE:[42] [465] Nocturnal 2097;  →  465.0
        #TITLE:[21] [260] Flaklypa;         →  260.0 (base BPM 130, 32nds)

    Args:
        lines: Líneas limpias del archivo .sm. / Clean lines from the .sm file.

    Returns:
        BPM como float, o None si el patrón no se encuentra.
        / BPM as float, or None if the pattern is not found.
    """

    title_payload = _extract_tag_payload(lines, "#TITLE")
    if not title_payload:
        return None

    match = _TITLE_BPM_RE.search(title_payload)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


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
            block = int(meter_text)
        except ValueError:
            block = 0

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
                "block": block,
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
    selected_block = -1

    for chart in charts:
        if chart.get("type") != "dance-single":
            continue
        difficulty = chart.get("difficulty", "")
        priority = _DIFFICULTY_PRIORITY.get(difficulty)
        if priority is None:
            continue

        block = chart.get("block", 0)
        if (
            selected is None
            or priority > selected_priority
            or (priority == selected_priority and block > selected_block)
        ):
            selected = chart
            selected_priority = priority
            selected_block = block

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


def _infer_subdivision(notes_per_measure: List[int]) -> int:
    """ES: Infiere la subdivisión real del chart a partir de la densidad de notas.

    Solo se consideran las subdivisiones válidas en stamina ITG: 16, 20, 24, 32.
    Se calcula la moda de notas por compás en compases de stream (≥ 12 notas,
    umbral holgado para captar incluso medidas parciales de stream) y se mapea
    a la subdivisión válida más cercana.

    EN: Infer the actual chart subdivision from note density.

    Only valid ITG stamina subdivisions are considered: 16, 20, 24, 32.
    The mode of notes-per-measure in stream-like measures (≥ 12 notes, a lenient
    threshold to catch even partial stream measures) is snapped to the nearest
    valid subdivision.

    Args:
        notes_per_measure: Conteo de notas por compás del parser.
                           / Per-measure note counts from the parser.

    Returns:
        Subdivisión válida más cercana (16, 20, 24 o 32).
        / Nearest valid subdivision (16, 20, 24 or 32).
    """

    # ES: Filtrar compases con al menos 12 notas (stream o cercano a stream).
    # EN: Filter measures with at least 12 notes (stream or near-stream).
    stream_counts = [n for n in notes_per_measure if n >= 12]
    if not stream_counts:
        return 16

    mode_count = Counter(stream_counts).most_common(1)[0][0]

    # ES: Mapear al valor válido más cercano: 16, 20, 24 o 32.
    # EN: Snap to the nearest valid value: 16, 20, 24 or 32.
    return min(_VALID_SUBDIVISIONS, key=lambda s: abs(s - mode_count))


def parse_sm_chart_with_meta(file_path: Path) -> Tuple[NotesData, int]:
    """ES: Parsea un .sm y devuelve notas por compás, Display BPM, block y subdivisión.

    El campo `block` en el dict resultante es la variable objetivo de clasificación,
    es decir, la dificultad numérica del chart (e.g. 19).

    EN: Parse an .sm and return per-measure notes, Display BPM, block, and subdivision.

    The `block` field in the returned dict is the classification target label:
    the numeric difficulty rating of the chart (e.g. 19).

    Args:
        file_path: Path to a StepMania .sm file.

    Returns:
        Tuple of (NotesData, subdivision) where NotesData contains `display_bpm`,
        `block` (the label to predict), and `notes_per_measure`.

    Raises:
        ValueError: If no suitable dance-single chart is found.
    """

    lines = _load_clean_lines(file_path)
    title_bpm = _extract_title_bpm(lines)
    timing_bpm = _extract_timing_bpm(lines)
    charts = _extract_charts(lines)
    chart = _select_hardest_chart(charts)

    measures = _prepare_measures(chart["note_lines"])
    densities, _row_counts = _count_notes_and_rows(measures)
    subdivision = _infer_subdivision(densities)
    notes_data: NotesData = {
        "display_bpm": title_bpm,
        "timing_bpm": timing_bpm,
        "block": chart["block"],
        "notes_per_measure": densities,
    }
    return notes_data, subdivision


def parse_sm_chart(file_path: Path) -> NotesData:
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
        Dict con `display_bpm`, `block` (variable objetivo de clasificación) y
        `notes_per_measure` para el chart seleccionado.
        / Dict with `display_bpm`, `block` (classification target label), and
        `notes_per_measure` for the selected chart.

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
