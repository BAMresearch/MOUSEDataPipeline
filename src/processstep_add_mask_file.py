from pathlib import Path
import subprocess

import h5py
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
WIP: This processing step finds the correct mask file for this measurement and adds it to the metadata
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    if not step_2_file.is_file():
        logger.info(f"Mask file determination not possible for {dir_path}, file missing at: {step_2_file}")
        return False

def get_configuration(filename: Path, logger: logging.Logger) -> int:
    """
    Read the configuration file and return the configuration
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            configuration = h5f['/entry1/instrument/configuration'][()]
    except Exception as e:
        logger.error(f"Error reading configuration from file: {e}")
        configuration = 0
    return configuration

from pathlib import Path
import logging
from datetime import datetime

def find_appropriate_mask(defaults: DefaultsCarrier, measurement_ymd: str, configuration: int, logger: logging.Logger) -> Path:
    """
    Finds the appropriate mask file based on measurement ymd and configuration.
    
    :param defaults: An instance of DefaultsCarrier containing default paths.
    :param measurement_ymd: `ymd` string of the measurement in format YYYYMMDD.
    :param configuration: Configuration number to match.
    :param logger: Logger object for logging purposes.
    :return: Path to the appropriate mask file.
    """
    mask_files = list(defaults.masks_dir.glob("*.nxs"))

    matching_masks = []

    # Extract ymd and configuration from each mask file
    for mask_file in mask_files:
        try:
            # Extract ymd from file name
            mask_ymd_str, mask_configuration_str = mask_file.stem.split('_')
            mask_ymd = datetime.strptime(mask_ymd_str, "%Y%m%d")
            mask_configuration = int(mask_configuration_str)

            # Check for matching configuration
            if mask_configuration == configuration:
                matching_masks.append((mask_file, mask_ymd))
        
        except Exception as e:
            logger.error(f"Error processing file {mask_file}: {e}")
    
    # Find the mask with the nearest `ymd` before or on measurement_ymd
    measurement_date = datetime.strptime(measurement_ymd, "%Y%m%d")
    best_mask = None
    smallest_difference = None

    for mask_file, mask_ymd in matching_masks:
        if mask_ymd <= measurement_date:
            difference = (measurement_date - mask_ymd).days
            if smallest_difference is None or difference < smallest_difference:
                smallest_difference = difference
                best_mask = mask_file

    if best_mask:
        logger.info(f"Selected mask file: {best_mask}")
    else:
        logger.warning(f"No suitable mask found for configuration {configuration} before {measurement_ymd}.")
    
    return best_mask


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    try:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
        input_file = dir_path / f'mouse_{ymd}_step_2.nxs'
        logger.info(f"Starting mask determination for {input_file}")
        configuration = get_configuration(input_file, logger)
        mask_file = find_appropriate_mask(defaults, ymd, configuration, logger)
        # write result to HDF5 file: 
        with h5py.File(input_file, 'a') as h5f:
            mask_file_dataset = h5f.require_dataset('/entry1/processing_metadata/mask_file', shape=(), dtype=h5py.special_dtype(vlen=str))
            mask_file_dataset[...] = str(mask_file)
        logger.info(f"Completed translator step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info("Processstep failed with stderr:")
        logger.info(e)
        logger.error(f"Error during translator subprocess: {e}")
