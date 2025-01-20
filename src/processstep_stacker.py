from pathlib import Path
import subprocess
from typing import Dict, List

import h5py
from YMD_class import YMD, extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

from processstep_add_mask_file import get_configuration

doc = """
WIP: This special processing step combines all repetitions in a batch. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False # we do this once per batch, so if we do it for one repetition, we don't need to do it again

def get_processed_files(dir_path: Path) -> List[Path]:
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    parent_path = dir_path.parent
    print(parent_path)
    processed_files = list(parent_path.glob(f'{ymd.YMD}_{batch}_*/MOUSE_{ymd.YMD}_{batch}_*.nxs'))
    return processed_files

def sort_processed_files_by_instrument_configuration(processed_files: List[Path], logger: logging.Logger) -> Dict[str, List[Path]]:
    """
    Sorts the processed files by instrument configuration, as read from the processed files themselves.
    Outputs a dictionary with the instrument configuration as the key and a sorted list of processed files as the value.
    """
    config_to_files: Dict[str, List[Path]] = {}

    for f in processed_files:
        measurement_config = str(get_configuration(f, logger))
        
        # Add the file to the dictionary under the correct key
        if measurement_config not in config_to_files:
            config_to_files[measurement_config] = []
        
        config_to_files[measurement_config].append(f)

    # Sort each list of files by their modification time (optional)
    for config, files in config_to_files.items():
        config_to_files[config] = sorted(files, key=lambda f: f.stat().st_mtime)

    return config_to_files

def processing_needed_for_config(dir_path: Path, ymd: YMD, batch: str, config: str, processed_files: List[Path], logger: logging.Logger) -> bool:
    parent_path = dir_path.parent
    stacked_file = parent_path / f'MOUSE_{ymd.YMD}_{batch}_{config}_stacked.nxs'  # Assuming a naming convention for the stacked file

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
    
    # default
    return False


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
    processed_files = get_processed_files(dir_path)
    files_by_config = sort_processed_files_by_instrument_configuration(processed_files, logger)

    for config in files_by_config.keys():
        if processing_needed_for_config(dir_path, ymd, batch, config, processed_files, logger):
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
        processed_files = get_processed_files(dir_path) # [str(f) for f in get_processed_files(dir_path)]
        files_by_config = sort_processed_files_by_instrument_configuration(processed_files, logger)

        # did I do this right?
        for config, files in files_by_config.items():
            files_as_str = [str(f) for f in files]
            stacked_file = parent_path / f'MOUSE_{ymd.YMD}_{batch}_{config}_stacked.nxs'  # Assuming a naming convention for the stacked file
            # output_file = dir_path.parent / 'translated.nxs'
            cmd = [
                'python3', str(pto_file),
                '-c', str(defaults.stacker_config_file),
                '-o', str(stacked_file),
                '-v', 
                # '-l',
                '-a', *files_as_str, # <-- processed files go here
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
