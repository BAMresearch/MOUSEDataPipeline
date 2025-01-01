from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
WIP: This processing step updates the metadata with the estimated thickness from the 
X-ray absorption and the X-ray absorption coefficient calculated from the composition. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beamanalysis not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    try:
        logger.info(f"Starting thickness_from_absorption step for {input_file}")
        # ...
        logger.info(f"Completed thickness_from_absorption step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info("thickness_from_absorption step failed with error:")
        logger.info(e)
        logger.error(f"Error during thickness_from_absorption step: {e}")