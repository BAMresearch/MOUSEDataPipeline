from pathlib import Path
import subprocess
from typing import Union

import h5py
import numpy as np
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement, LinkElement
from HDF5Translator.translator import process_translation_element, process_link_element
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
    # deal with absorption < 0, which can happen due to noise in the transmission measurement for very high transmissions (e.g. vacuum measurements)
    abs_sign = np.sign(absorption)  # we calculate thicknesses with negative absorption as "negative thickness", so that the average for multiple repetitions is not biased
    abs_val = abs(absorption)  # we only use the absolute value for the calculation
    # we cannot deal with absorption > 1, which can happen due to noise in the transmission measurement for extremely low transmissions (e.g. very thick samples). This would make np.log(1-abs) imaginary

    if absorption_coefficient == 0:
        logger.warning(f'absorption coefficient is zero, cannot calculate thickness')
        return -1
    if not (0 < abs_val <= 1):
        logger.warning(f'absorption value {abs_val} is not in the range [0, 1]')
        return -1
    thickness = -1 * abs_sign * np.log(1-abs_val) / absorption_coefficient
    return thickness


def get_background_file(filename: Path, logger: logging.Logger) -> Union[Path, None]:
    """
    Returns the background file for a given sample.
    """
    background_file = get_str_from_h5(filename, '/entry1/processing_required_metadata/background_file', logger)
    # make it relative to the current file
    if background_file:
        background_file = (filename.parent / background_file).resolve()
    # print(f' * * * Found background file {background_file} for sample file {filename}')
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
        # collect absorption coefficient and transmission factor from the file
        absorption_coefficient = get_float_from_h5(input_file, HDFPath='/entry1/sample/overall_mu', logger=logger)
        if absorption_coefficient <= 0:
            logger.warning(f'absorption coefficient negative or zero in {input_file}, cannot calculate thickness')
            return
        transmission = get_float_from_h5(input_file, HDFPath='/entry1/sample/transmission', logger=logger)
        absorption = 1 - transmission
        # we also need to get the absorption from the background file if it exists: 
        background_file = get_background_file(input_file, logger)

        # get the transmission from the background file if we can, to later isolate the absorption from the sample only
        absorption_bg = 0
        absorption_sample = absorption
        if background_file:
            bg_ymd, bg_batch, _ = background_file.stem.split('_')[1:4]
            # make integers:
            bg_ymd, bg_batch = int(bg_ymd), int(bg_batch)
            if (int(batch) == bg_batch) and (ymd.as_int() == bg_ymd):
                logger.info(f"Sample and background file are the same: {input_file} and {background_file}")
                logger.info("Not correcting for background transmission")
            else:
                # we can get the mean from the stacked background file. transmission will be meaned if it is an array by get_float_from_h5
                transmission_bg = get_float_from_h5(background_file, HDFPath='/entry1/sample/transmission', logger=logger)
                # here we assume that the background absorption is only due to the container
                transmission_sample = transmission / transmission_bg  # if transmission_bg >= 0 else transmission # nope, have to account for noise and rely on averaging. 
                absorption_sample = 1-transmission_sample
                # if not (0 < absorption_sample < 1):
                #     logger.warning(f"Sample-specific absorption {absorption_sample} outside of realistic limits. total absorption: {absorption}, background absorption: {absorption_bg}. resetting to {absorption}")
                #     absorption_sample = absorption

        # Calculate the thickness from the absorption data
        thickness = calculate_thickness(absorption_coefficient, absorption_sample, logger)
        logging.info(f'Calculated {thickness=:0.03e} m from {absorption=:0.03e} ({absorption_sample=:0.03e} and background absorption {absorption_bg=:0.03e} from {background_file=}) and {absorption_coefficient=:0.03e} 1/m for file {input_file})')

        # Now let's store all that information in the HDF5 file. 
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

        # lastly select the appropriate thickness to be used in the calculations:
        # if the thickness specified in the logbook is negative, we use the absorption-derived thickness,
        # but if it is positive, we use the logbook-specified thickness
        # we set the /entry1/sample/thickness to /entry1/sample/samplethickness if not negative, else set to absorptionDerivedThickness
        samplethickness = get_float_from_h5(input_file, HDFPath='/entry1/sample/samplethickness', logger=logger)
        if samplethickness < 0:
            logger.info(f'setting samplethickness to absorptionDerivedThickness since logbook-specified samplethickness was {samplethickness}')

            TElements += [TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/thickness",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=thickness,
                source_units="m",
                destination_units="m",
                attributes={
                    "note": "Set to absorptionDerivedThickness since logbook-specified samplethickness was negative. Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            )]
            # "/entry1 /sample/absorptionDerivedThickness"

            # LE = LinkElement(
            #     source_path="/entry1/sample/absorptionDerivedThickness",
            #     destination_path="/entry1/sample/thickness",
            #     soft_or_hard_link="hard",
            # )
        else:
            logger.info(f'keeping logbook-specified samplethickness {samplethickness} since it was positive')
            TElements += [TranslationElement(
                # source is none since we're storing derived data
                destination="/entry1/sample/thickness",
                minimum_dimensionality=1,
                data_type="float32",
                default_value=samplethickness,
                source_units="m",
                destination_units="m",
                attributes={
                    "note": "Set to logbook-specified samplethickness since it was positive. Determined by the processstep_thickness_from_absorption post-translation processing script."
                },
            )]
        
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
