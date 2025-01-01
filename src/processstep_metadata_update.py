from pathlib import Path
import subprocess
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
WIP: This processing step updates the metadata in the translated and beam-analyzed files 
with details from the logbook and project/sample information
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run.
    """
    eiger_file = dir_path / 'im_craw.nxs'
    if eiger_file.exists():
        logger.debug(f"Translator step can run: Found {eiger_file}")
        return True
    logger.debug(f"Translator step skipped: No im_craw.nxs in {dir_path}")
    return False


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
