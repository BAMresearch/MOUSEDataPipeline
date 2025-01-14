from pathlib import Path
import subprocess
from typing import List
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
WIP: This special processing step combines all repetitions in a batch. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False # we do this once per batch, so if we do it for one repetition, we don't need to do it again

def get_processed_files(dir_path: Path) -> List[Path]:
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    parent_path = dir_path.parent
    print(parent_path)
    processed_files = list(parent_path.glob(f'{ymd.YMD}_{batch}_*/mouse_{ymd.YMD}_step_2.nxs'))
    return processed_files

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run. Here, we need to do four things: 
    0) check if there is a stacked file already, if not, we need to run this step.
    1) use Path.glob to find all the processed files of a batch. 
    2) check what the latest processed file of this list has as a modification date. 
    3) check if the date of that latest processed file is newer than the date of the stacked file this process produces. 
    If 0 or 3 are true, we need to run this step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    parent_path = dir_path.parent
    stacked_file = parent_path / f'{ymd.YMD}_{batch}_stacked.nxs'  # Assuming a naming convention for the stacked file
    processed_files = get_processed_files(dir_path)

    if not processed_files:
        logger.info(f"No processed files found for batch in {dir_path}, cannot run")
        return False

    if not stacked_file.is_file():
        logger.info(f"Stacked file not found, processing needed for {dir_path}")
        return True

    latest_processed_file = max(processed_files, key=lambda f: f.stat().st_mtime)

    if latest_processed_file.stat().st_mtime > stacked_file.stat().st_mtime:
        logger.info(f"Processed file {latest_processed_file} is newer than stacked file, processing needed for {dir_path}")
        return True

    # If none of the conditions necessitate processing, return False
    logger.info(f"No need to rerun processing for {dir_path}")
    return False


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    try:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
        parent_path = dir_path.parent
        pto_file = defaults.post_translation_dir / 'post_translation_operation_hdf5_stacker.py'
        stacked_file = parent_path / f'{ymd.YMD}_{batch}_stacked.nxs'  # Assuming a naming convention for the stacked file
        processed_files = [str(f) for f in get_processed_files(dir_path)]
        # output_file = dir_path.parent / 'translated.nxs'
        cmd = [
            'python3', str(pto_file),
            '-c', str(defaults.stacker_config_file),
            '-o', str(stacked_file),
            '-v', 
            '-l',
            '-a', *processed_files, # <-- processed files go here
        ]
        logger.info(f"Starting stacker step for {parent_path}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed stacker step for {parent_path}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        # Optionally, also print the standard output
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during stacker subprocess: {e}")
        raise
