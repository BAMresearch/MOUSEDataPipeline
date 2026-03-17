import logging
import shutil
import subprocess
import sys
from pathlib import Path

import h5py

from defaults_carrier import DefaultsCarrier
from logbook_support import LogbookReaderLike
from YMD_class import YMD, extract_metadata_from_path

doc = """
Update translated measurement files with logbook, project, and sample metadata
using the mouse_logbook CLI writer.
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


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
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if e.stdout and e.stdout.strip():
            logger.error("mouse_logbook stdout for %s:\n%s", input_file, e.stdout.strip())
        if e.stderr and e.stderr.strip():
            logger.error("mouse_logbook stderr for %s:\n%s", input_file, e.stderr.strip())
        logger.error("Metadata update failed for %s with command: %s", input_file, " ".join(command))
        raise
    if result.stdout.strip():
        logger.info(result.stdout.strip())
    if result.stderr.strip():
        logger.info(result.stderr.strip())


def _ensure_sampleowner_compatibility(output_file: Path, logger: logging.Logger) -> None:
    with h5py.File(output_file, "a") as h5f:
        if "/entry1/sample/sampleowner" in h5f:
            return
        if "/entry1/sample/owner" not in h5f:
            logger.warning("mouse_logbook did not write /entry1/sample/owner in %s", output_file)
            return

        owner_dataset = h5f["/entry1/sample/owner"]
        owner_value = owner_dataset[()]
        sampleowner_dataset = h5f.require_dataset(
            "/entry1/sample/sampleowner",
            shape=owner_dataset.shape,
            dtype=owner_dataset.dtype,
        )
        sampleowner_dataset[...] = owner_value
        sampleowner_dataset.attrs["note"] = "Compatibility alias copied from /entry1/sample/owner."


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
    _ensure_sampleowner_compatibility(input_file, logger)
    logger.info(f"Completed metadata update via mouse_logbook CLI for {input_file}")
