from pathlib import Path
import subprocess

import h5py
import numpy as np
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

doc = """
This processing step updates the metadata with the estimated thickness from the 
X-ray absorption and the X-ray absorption coefficient calculated from the composition. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beamanalysis not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True

def get_absorption(filename: Path, logger: logging.Logger) -> float:
    """
    Returns the absorption value from the file.
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            transmission = h5f['/entry1/sample/transmission'][()].mean()
    except Exception as e:
        logger.warning(f'could not read transmission from {filename} with error {e}')
        return 0.0
    if not isinstance(transmission, np.floating):
        logger.warning(f'transmission not found in file {filename}')
        return 0.0
    if not (0 < transmission <= 1):
        logger.warning(f'transmission value {transmission} is not in the range [0, 1]')
        return 0.0
    return 1 - transmission

def get_absorption_coefficient(filename: Path, logger: logging.Logger) -> float:
    """
    Returns the X-ray absorption coefficient (in 1/m) for the sample and energy from the file.
    """
    try: 
        with h5py.File(filename, 'r') as h5f:
            absorption_coefficient = h5f['/entry1/sample/overall_mu'][()]
    except Exception as e:
        logger.warning(f'could not read absorption coefficient from {filename} with error {e}')
        return 0.0
    if not isinstance(absorption_coefficient, np.floating):
        logger.warning(f'absorption coefficient not found in file {filename}')
        return 0.0
    if absorption_coefficient <= 0:
        logger.warning(f'absorption coefficient negative or zero in {filename}')
        return 0.0
    return absorption_coefficient

def calculate_thickness(absorption_coefficient: float, absorption: float, logger: logging.Logger) -> float:
    """
    Calculates the thickness of the sample from the absorption and the absorption coefficient.
    """
    if absorption_coefficient == 0:
        logger.warning(f'absorption coefficient is zero, cannot calculate thickness')
        return -1
    if not (0 < absorption <= 1):
        logger.warning(f'absorption value {absorption} is not in the range [0, 1]')
        return -1
    thickness = -1 * np.log10(absorption) / absorption_coefficient
    return thickness


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    try:
        logger.info(f"Starting thickness_from_absorption step for {input_file}")
        absorption_coefficient = get_absorption_coefficient(input_file, logger)
        absorption = get_absorption(input_file, logger)
        thickness = calculate_thickness(absorption_coefficient, absorption, logger)
        with h5py.File(input_file, 'a') as h5f:
            # this will be stacked into a list of thicknesses... 
            tloc = h5f.require_dataset('/entry1/sample/absorptionDerivedThicknesses', shape=(), dtype=np.float32)
            tloc[...] = thickness
            tloc.attrs['units'] = 'm'

        logger.info(f"Completed thickness_from_absorption step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info("thickness_from_absorption step failed with error:")
        logger.info(e)
        logger.error(f"Error during thickness_from_absorption step: {e}")