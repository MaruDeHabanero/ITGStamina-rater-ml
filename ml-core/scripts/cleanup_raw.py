"""
English: Remove every file that is not a .sm or .ssc chart inside the raw data folder.
Español: Elimina todos los archivos que no sean charts .sm o .ssc dentro de la carpeta de datos crudos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

ALLOWED_EXTENSIONS = {".sm", ".ssc"}


def list_extra_files(raw_root: Path) -> List[Path]:
    """
    English: Collect files under raw_root that are not .sm or .ssc charts.
    Español: Recolecta los archivos dentro de raw_root que no son charts .sm o .ssc.
    """

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw path is not a directory: {raw_root}")

    extra_files: List[Path] = []
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() not in ALLOWED_EXTENSIONS:
            extra_files.append(path)
    return extra_files


def delete_files(paths: Iterable[Path]) -> None:
    """
    English: Delete the provided files.
    Español: Elimina los archivos proporcionados.
    """

    for file_path in paths:
        file_path.unlink(missing_ok=True)


def collect_file_stats(root: Path) -> Tuple[int, int]:
    """
    English: Count files and total bytes under root.
    Español: Cuenta archivos y bytes totales dentro de root.
    """

    total_files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            total_files += 1
            total_bytes += path.stat().st_size
    return total_files, total_bytes


def format_size(bytes_count: int) -> str:
    """
    English: Render bytes as a human-friendly string (two decimals).
    Español: Muestra bytes en un formato legible (dos decimales).
    """

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_count)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} {units[-1]}"


def write_report(
    report_path: Path,
    *,
    raw_root: Path,
    total_before: int,
    total_after: int,
    bytes_before: int,
    bytes_after: int,
    deleted_files: List[Path],
) -> None:
    """
    English: Persist a cleanup changelog with sizes and file counts.
    Español: Guarda un registro de limpieza con tamaños y conteos de archivos.
    """

    report_path.parent.mkdir(parents=True, exist_ok=True)
    relative_deleted = [f.relative_to(raw_root) for f in deleted_files]

    lines = [
        f"Raw root: {raw_root}",
        f"Space before deletion: {format_size(bytes_before)} ({bytes_before} bytes)",
        f"Space after deletion: {format_size(bytes_after)} ({bytes_after} bytes)",
        f"Total files before deletion: {total_before}",
        f"Files deleted: {len(deleted_files)}",
        f"Total files after deletion: {total_after}",
        "Deleted files:",
    ]

    if relative_deleted:
        lines.extend(str(path) for path in relative_deleted)
    else:
        lines.append("(none)")

    report_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    """
    English: Parse CLI arguments for cleanup automation.
    Español: Analiza los argumentos de línea de comandos para la automatización de limpieza.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Remove every non-.sm/.ssc file inside ml-core/data/raw. "
            "Uses a dry-run by default to preview deletions."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help=(
            "English: Custom raw data root. "
            "Español: Ruta personalizada a la carpeta raw."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "English: Show files that would be removed without deleting them. "
            "Español: Muestra los archivos a eliminar sin borrarlos."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "cleanup_report.txt",
        help=(
            "English: Where to write the cleanup report. "
            "Español: Ruta donde se guardará el informe de limpieza."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    English: Execute the cleanup workflow (list extras, optionally delete).
    Español: Ejecuta el flujo de limpieza (lista extras y opcionalmente borra).
    """

    args = parse_args()
    raw_root: Path = args.raw_root
    report_path: Path = args.report_path

    extra_files = list_extra_files(raw_root)

    if not extra_files:
        print("No extra files found. Nothing to delete.")
        return

    print("Files marked for deletion:")
    for file_path in extra_files:
        print(f"- {file_path}")

    if args.dry_run:
        print("Dry run enabled. No files were deleted.")
        return

    total_before, bytes_before = collect_file_stats(raw_root)
    delete_files(extra_files)
    total_after, bytes_after = collect_file_stats(raw_root)

    write_report(
        report_path,
        raw_root=raw_root,
        total_before=total_before,
        total_after=total_after,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        deleted_files=extra_files,
    )

    print(f"Deleted {len(extra_files)} file(s). Report saved to {report_path}.")


if __name__ == "__main__":
    main()
