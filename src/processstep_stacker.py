from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
WIP: This special processing step combines all repetitions in a batch. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beamanalysis not possible for {dir_path}, result file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    try:
        input_file = dir_path / 'im_craw.nxs'
        output_file = dir_path / 'translated.nxs'
        cmd = [
            'python3', '-m', 'HDF5Translator',
            '-C', str(defaults.translator_config),
            '-I', str(input_file),
            '-O', str(output_file)
        ]
        logger.info(f"Starting translator step for {input_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed translator step for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        # Optionally, also print the standard output
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during translator subprocess: {e}")
        raise
