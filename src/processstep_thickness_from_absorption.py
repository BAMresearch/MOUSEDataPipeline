from pathlib import Path
import subprocess
from typing import Union

import h5py
import numpy as np
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from utilities import get_float_from_h5, get_str_from_h5

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
    thickness = -1 * np.log(1-absorption) / absorption_coefficient
    return thickness


def get_background_file(filename: Path, logger: logging.Logger) -> Union[Path, None]:
    """
    Returns the background file for a given sample.
    """
    background_file = get_str_from_h5(filename, '/entry1/processing_required_metadata/background_file', logger)
    # make it relative to the current file
    if background_file:
        background_file = (filename.parent / background_file).resolve()
    if background_file and Path(background_file).is_file():
        return Path(background_file)
    else:
        return None


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    assert input_file.is_file(), f"Input file {input_file} does not exist"

    try:
        logger.info(f"Starting thickness_from_absorption step for {input_file}")
        absorption_coefficient = get_float_from_h5(input_file, HDFPath='/entry1/sample/overall_mu', logger=logger)
        if absorption_coefficient <= 0:
            logger.warning(f'absorption coefficient negative or zero in {input_file}, cannot calculate thickness')
            return
        transmission = get_float_from_h5(input_file, HDFPath='/entry1/sample/transmission', logger=logger)
        absorption = 1 - transmission
        # we also need to get the absorption from the background file if it exists: 
        background_file = get_background_file(input_file, logger)

        absorption_bg = 0
        absorption_sample = absorption
        if background_file:
            bg_ymd, bg_batch, _ = background_file.stem.split('_')[1:4]
            if (batch == bg_batch) and (ymd == bg_ymd):
                logger.warning(f"Sample and background file are the same: {input_file} and {background_file}")
                logger.warning(f"Not correcting for background transmission")
            else:
                # we can get the mean from the stacked background file
                transmission_bg = get_float_from_h5(background_file, HDFPath='/entry1/sample/transmission_mean', logger=logger)
                transmission_sample = transmission / transmission_bg if transmission_bg > 0 else transmission
                absorption_sample = 1-transmission_sample
                if not(0 < absorption_sample < 1):
                    logger.warning(f"Sample-specific absorption {absorption_sample} outside of realistic limits. total absorption: {absorption}, background absorption: {absorption_bg}. resetting to {absorption}")
                    absorption_sample = absorption
                
        thickness = calculate_thickness(absorption_coefficient, absorption_sample, logger)

        # This class lets you configure exactly what the output should look like in the HDF5 file.
        TElements = []  # we want to add two elements, so I make a list
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/absorptionDerivedThickness",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=thickness,
                source_units="m",
                destination_units="m",
                attributes={
                    "note": "Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/absorption_total",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=absorption,
                source_units="",
                destination_units="",
                attributes={
                    "note": "Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/absorption_by_sample",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=absorption_sample,
                source_units="",
                destination_units="",
                attributes={
                    "note": "Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/absorption_by_bg",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=absorption_bg,
                source_units="",
                destination_units="",
                attributes={
                    "note": "Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            ),
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)

        logger.info(f"Completed thickness_from_absorption step for {input_file}, {absorption=}, {absorption_bg=}. {absorption_sample=}, {absorption_coefficient=}, {thickness=}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info("thickness_from_absorption step failed with error:")
        logger.info(e)
        logger.error(f"Error during thickness_from_absorption step: {e}")

