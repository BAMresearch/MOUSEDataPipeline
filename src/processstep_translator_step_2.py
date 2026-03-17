import logging
import subprocess
from pathlib import Path

from defaults_carrier import DefaultsCarrier
from logbook_support import LogbookReaderLike
from YMD_class import extract_metadata_from_path

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def can_run(
    dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger
) -> bool:
    """
    Checks if the translator step should run. Besides the base files, we don't need anything...
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_1_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs"
    if not step_1_file.is_file():
        logger.info(f"Step 2 translation not possible for {dir_path}, step 1 result file missing at: {step_1_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger):
    """
    Executes the second translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:
        del logbook_reader
        input_file = next(dir_path.glob("eiger_*_master.h5"), None)
        template_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs"
        output_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}.nxs"
        cmd = [
            "python3",
            "-m",
            "HDF5Translator",
            "-C",
            str(defaults.translator_template_dir / "BAM_new_MOUSE_dectris_adder_configuration.yaml"),
            "-T",
            str(template_file),
            "-I",
            str(input_file),
            "-O",
            str(output_file),
            "-d",
        ]
        logger.info(f"Starting translator step 2 for {input_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed translator step 2 for {input_file}")
    except subprocess.CalledProcessError as e:
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during translator subprocess: {e}")
        raise
