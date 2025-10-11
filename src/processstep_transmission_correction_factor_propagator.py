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
We also propagate the scattering probability estimate to all files in the batch, as this is also distance independent.
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if this step can run. We only want to run this once per batch, so we check if we are in the lowest repetition
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    available_files = get_processed_files(dir_path)
    # get the lowest repetition from the available files
    if not available_files:
        logger.info(f"No processed files found for batch in {dir_path}, cannot run")
        return False
    lowest_repetition = min([extract_metadata_from_path(f)[2] for f in available_files])
    if repetition != lowest_repetition:
        logger.info(f"Not the lowest repetition for batch in {dir_path}, cannot run")
        return False
    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the processing step.
    This step first reads the transmission correction factor for all configurations. From these, it picks the largest factor (i.e. with the most collected scattering/diffraction))
    Finally, it updates the metadata in all files of the batch with this correction factor.
    """
    try:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
        available_files = get_processed_files(dir_path)
        if not available_files:
            logger.info(f"No processed files found for batch in {dir_path}, cannot run")
            return
        # get the largest correction factor from the files
        largest_correction_factor = None
        largest_scattering_probability = None
        for f in available_files:
            correction_factor = get_float_from_h5(f, "/entry1/sample/transmission_correction_factor", logger)
            scattering_probability = get_float_from_h5(f, "/entry1/sample/scattering_probability_estimate", logger)
            if correction_factor is not None:
                logger.info(f"Found correction factor for {f}: {correction_factor}")
                # Update the largest correction factor if needed
                if largest_correction_factor is None or correction_factor > largest_correction_factor:
                    largest_correction_factor = correction_factor
            if scattering_probability is not None:
                logger.info(f"Found scattering probability for {f}: {scattering_probability}")
                # Update the largest scattering probability if needed
                if largest_scattering_probability is None or scattering_probability > largest_scattering_probability:
                    largest_scattering_probability = scattering_probability


        if largest_correction_factor == 0.0 or largest_correction_factor is None or largest_scattering_probability is None:
            logger.info(f"No valid correction factors found for batch in {dir_path}, cannot run")
            return

        # propagate this correction factor to all files in the batch
        for f in available_files:
            with h5py.File(f, 'a') as h5f:
                h5f.create_dataset("/entry1/sample/largest_transmission_correction_factor", data=largest_correction_factor)
                h5f.create_dataset("/entry1/sample/largest_scattering_probability_estimate", data=largest_scattering_probability)
            logger.debug(f"Updated correction factor for {f} to {largest_correction_factor}")
        logger.info(f"Completed correction factor propagation for batch in {dir_path}, using factor: {largest_correction_factor}")

    except Exception as e:
        # Print the standard output and standard error
        logger.info("correction factor propagation step failed with error:")
        logger.info(e)
        logger.error(f"Error during correction factor propagation step: {e}")