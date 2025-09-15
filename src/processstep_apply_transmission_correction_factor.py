from pathlib import Path
# import subprocess

import h5py
import numpy as np
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
import csv
from processstep_stacker import get_processed_files
from utilities import get_float_from_h5, get_str_from_h5
doc = """
This processing step updates the metadata transmission_correction_factor with the correction factor from the closest distance. 
This correction factor approximates the correction for adding the scattering to the transmitted beam. It's not perfect, but at the moment the best we can do. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if this step can run. We only want to run this once per batch, so we check if we are in the lowest repetition
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"metadata_updater cannot run in {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the processing step.
    This step applies the transmission correction factor to the transmission factor stored in the same file.
    """
    try:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'

        # copy the old transmission value to a new field transmission_beam
        with h5py.File(input_file, "r+") as h5_out:
            h5_out['/entry1/sample/transmission_beam'] = h5_out['/entry1/sample/transmission'][...]

        transmission = get_float_from_h5(input_file, '/entry1/sample/transmission', logger)
        transmission_correction_factor = get_float_from_h5(input_file, '/entry1/sample/transmission_correction_factor', logger)
        if transmission_correction_factor and transmission_correction_factor > 1.0:
            transmission *= transmission_correction_factor

        TE = TranslationElement(
                destination='/entry1/sample/transmission',
                data_type="float",
                minimum_dimensionality=1,
                default_value=transmission,
                attributes={
                    "note": "Beam transmission value corrected with the transmission_correction_factor to approximate the total transmission including scattered/diffracted photons.",
                },
            )

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            process_translation_element(None, h5_out, TE)

    except Exception as e:
        # Print the standard output and standard error
        logger.info("correction factor propagation step failed with error:")
        logger.info(e)
        logger.error(f"Error during correction factor propagation step: {e}")