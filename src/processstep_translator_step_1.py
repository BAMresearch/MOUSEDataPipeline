import logging
import subprocess
import sys
from pathlib import Path

from checkers import processing_possible
from defaults_carrier import DefaultsCarrier
from logbook_support import LogbookReaderLike
from YMD_class import extract_metadata_from_path

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def _is_step_1_output_up_to_date(dir_path: Path, defaults: DefaultsCarrier) -> bool:
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / "im_craw.nxs"
    output_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs"
    config_file = defaults.translator_template_dir / "BAM_new_MOUSE_xenocs_translator_configuration.yaml"
    if not output_file.is_file():
        return False
    output_mtime = output_file.stat().st_mtime
    return output_mtime >= max(input_file.stat().st_mtime, config_file.stat().st_mtime)


def can_run(
    dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger
) -> bool:
    """
    Checks if the translator step should run. Besides the base files, we don't need anything...
    """
    missing_files = processing_possible(dir_path, return_list=True)
    if missing_files:
        logger.info(
            f"Step 1 translation not possible for {dir_path}, required files missing... check DEBUG for details"
        )
        logger.debug(
            f"Required files missing for step 1 translation not possible for {dir_path}. The following were not found: {missing_files}"
        )
        return False
    if _is_step_1_output_up_to_date(dir_path, defaults):
        logger.info("Step 1 translation already up to date for %s", dir_path)
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger):
    """
    Executes the first translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:
        # encode: python3 -m HDF5Translator -C BAM_new_MOUSE_xenocs_translator_configuration.yaml -I ./20250101_17_0/im_craw.nxs -O ./20250101_17_0/testBAM.nxs -d

        input_file = dir_path / "im_craw.nxs"
        output_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs"
        cmd = [
            sys.executable,
            "-m",
            "HDF5Translator",
            "-C",
            str(defaults.translator_template_dir / "BAM_new_MOUSE_xenocs_translator_configuration.yaml"),
            "-I",
            str(input_file),
            "-O",
            str(output_file),
            "-d",
        ]
        logger.info(f"Starting translator step 1 for {input_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed translator step 1 for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        # Optionally, also print the standard output
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during translator subprocess: {e}")
        raise
