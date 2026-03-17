from pathlib import Path
import logging
import shutil
import subprocess
import sys

from defaults_carrier import DefaultsCarrier
from logbook_support import LogbookReaderLike
from YMD_class import YMD, extract_metadata_from_path

doc = """
Update translated measurement files with logbook, project, and sample metadata
using the mouse_logbook CLI writer.
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False


def can_run(
    dir_path: Path,
    defaults: DefaultsCarrier,
    logbook_reader: LogbookReaderLike | None,
    logger: logging.Logger,
) -> bool:
    """
    Checks whether the translated NeXus file exists.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    output_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}.nxs"
    if not output_file.is_file():
        logger.info(f"metadata_updater cannot run in {dir_path}, file missing at: {output_file}")
        return False
    return True


def _resolve_mouse_logbook_cli() -> Path:
    candidates = [
        shutil.which("mouse-logbook"),
        str(Path(sys.executable).resolve().with_name("mouse-logbook")),
        str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "mouse-logbook"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError("Could not locate a mouse-logbook executable.")


def _run_mouse_logbook_cli(
    input_file: Path,
    ymd: YMD,
    batch: int,
    defaults: DefaultsCarrier,
    logger: logging.Logger,
) -> None:
    cli = _resolve_mouse_logbook_cli()
    command = [
        str(cli),
        "write-nexus-metadata",
        str(defaults.logbook_file),
        str(defaults.projects_dir),
        str(input_file),
        "--ymd",
        ymd.YMD,
        "--batch-num",
        str(batch),
    ]
    logger.info("Running metadata update via mouse_logbook CLI: %s", " ".join(command))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stdout.strip():
        logger.info(result.stdout.strip())
    if result.stderr.strip():
        logger.info(result.stderr.strip())


def run(
    dir_path: Path,
    defaults: DefaultsCarrier,
    logbook_reader: LogbookReaderLike | None,
    logger: logging.Logger,
):
    """
    Update metadata through the mouse_logbook CLI.
    """
    del logbook_reader
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}.nxs"
    _run_mouse_logbook_cli(
        input_file=input_file,
        ymd=ymd,
        batch=batch,
        defaults=defaults,
        logger=logger,
    )
    logger.info(f"Completed metadata update via mouse_logbook CLI for {input_file}")
