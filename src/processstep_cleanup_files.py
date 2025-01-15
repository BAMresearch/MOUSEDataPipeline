from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
This processing step cleans up the temporary and intermediate files created during the processing
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run.
    """
    return True # can always run this step

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_1_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs'
    # add any other temporary files that might be created during processing

    try:
        filenames = [
            step_1_file
        ]
        cmd = [
            'rm', '-f', *filenames
        ]
        logger.info(f"Starting cleanup step for {ymd=}, {batch=}, {repetition=}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed cleanup step for {ymd=}, {batch=}, {repetition=}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        # Optionally, also print the standard output
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during cleanup subprocess: {e}")
        raise
