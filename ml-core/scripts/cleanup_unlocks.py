from __future__ import annotations

"""
ES: Limpia charts de "Unlocks" que no son stamina según el banner.
EN: Remove non-stamina "Unlocks" charts based on #BANNER metadata.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# Banners que identifican charts de tech/no-stamina en los packs "- Unlocks".
TECH_BANNERS = {
    "srpg5zdprtunlockbn.png",
    "srpg6zdprtunlockbn.png",
    "srpg7bdprtunlockbn.png",
    "srpg8bdprtunlockbn.png",
}


@dataclass
class ChartCandidate:
    """ES: Datos para un chart a eliminar. EN: Data for a chart candidate."""

    folder: Path
    sm_file: Path
    banner: str
    size_bytes: int


def find_sm_files(chart_folder: Path) -> List[Path]:
    """ES: Busca archivos .sm en la carpeta del chart. EN: Find .sm files."""

    return [p for p in chart_folder.glob("*.sm") if p.is_file()]


def read_banner(sm_path: Path) -> Optional[str]:
    """ES: Lee el valor de #BANNER en un .sm. EN: Read #BANNER from .sm."""

    try:
        with sm_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("#BANNER:"):
                    value = line[len("#BANNER:") :].strip()
                    # Quita el ';' final si existe.
                    return value.rstrip(";\n\r")
    except OSError:
        return None
    return None


def dir_size_bytes(path: Path) -> int:
    """ES: Calcula el tamaño total en bytes de un directorio. EN: Dir size."""

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def find_tech_candidates(root: Path) -> List[ChartCandidate]:
    """ES: Encuentra charts no-stamina en carpetas "- Unlocks". EN: Find tech."""

    candidates: List[ChartCandidate] = []

    for unlock_dir in root.glob("* - Unlocks"):
        if not unlock_dir.is_dir():
            continue
        for chart_dir in unlock_dir.iterdir():
            if not chart_dir.is_dir():
                continue
            sm_files = find_sm_files(chart_dir)
            if not sm_files:
                continue
            for sm_path in sm_files:
                banner = read_banner(sm_path)
                if banner and banner in TECH_BANNERS:
                    candidates.append(
                        ChartCandidate(
                            folder=chart_dir,
                            sm_file=sm_path,
                            banner=banner,
                            size_bytes=dir_size_bytes(chart_dir),
                        )
                    )
                    break  # Una coincidencia por carpeta es suficiente. Solamente deben de tener 1 archivo .sm.
    return candidates


def human_size(nbytes: int) -> str:
    """ES/EN: Tamaño amigable."""

    step = 1024.0
    if nbytes >= step ** 3:
        return f"{nbytes / (step ** 3):.2f} GiB"
    if nbytes >= step ** 2:
        return f"{nbytes / (step ** 2):.2f} MiB"
    if nbytes >= step:
        return f"{nbytes / step:.2f} KiB"
    return f"{nbytes} B"


def delete_candidates(candidates: Iterable[ChartCandidate], apply: bool) -> None:
    """ES: Borra (o simula) carpetas de charts tech. EN: Delete/simulate."""

    total_bytes = sum(c.size_bytes for c in candidates)
    print("= Unlocks tech cleanup =")
    print(f"Charts marcados (tech): {len(list(candidates))}")
    print(f"Espacio estimado: {human_size(total_bytes)}")

    for cand in candidates:
        rel = cand.folder
        print(f"- {rel} | banner={cand.banner} | size={human_size(cand.size_bytes)}")
        if apply:
            try:
                shutil.rmtree(cand.folder)
            except OSError as exc:
                print(f"  ! No se pudo borrar {rel}: {exc}")

    if apply:
        print("Borrado completado.")
    else:
        print("Dry-run: no se borró nada.")


def main() -> None:
    """ES: CLI para limpiar charts tech en "- Unlocks". EN: CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Remove tech charts from Unlocks packs")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Root folder containing packs (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete anything; just list candidates",
    )

    args = parser.parse_args()

    candidates = find_tech_candidates(args.root)
    delete_candidates(candidates, apply=not args.dry_run)


if __name__ == "__main__":
    main()