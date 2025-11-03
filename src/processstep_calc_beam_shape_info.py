from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader  # type: ignore
import logging
from processstep_calc_beam_flux_and_transmissions import dynamic_beam_analysis
from utilities import prepare_eiger_image
import h5py
from HDF5Translator.translator_elements import TranslationElement  # type: ignore
from HDF5Translator.translator import process_translation_element  # type: ignore
from utilities import get_float_from_h5

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the beam information can run. We need the translated file.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beam information not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    After the beam center and beam masks have been determined, we can optionally get some extra information
    on the beam shape. This includes the beam widths (sigma minor, sigma major), and the angle theta of the major axis.
    """
    DirectBeamDatapath = "/entry1/processing/direct_beam_profile/data"
    SigmaMinorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_minor"
    SigmaMajorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_major"
    ThetaOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/theta"

    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:

        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'

        logger.info(f"Starting beam info determination for {input_file}")

        with h5py.File(input_file, "r") as h5_in:
            # Read necessary information (this is just a placeholder, adapt as needed)
            DirectBeamData = prepare_eiger_image(h5_in[DirectBeamDatapath][()], logger)

        # compute the needed values:
        _, _, _, _, sigma_minor, sigma_major, theta, _ = dynamic_beam_analysis(
            DirectBeamData, coverage=0.997, beam_coverage_mask=None
        )
        # print(f'{repetition=}, {ITotal_overall/DirectBeamDuration=:0.02f}, {ITotal_region/DirectBeamDuration=:0.02f} int over {DirectBeamDuration=:0.02f}s')

        # Write out the beam sigmas and theta:
        TElements = []  # we want to add two elements, so I make a list
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=SigmaMinorOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=sigma_minor,
                source_units="px",
                destination_units="px",
                attributes={
                    "note": "Sigma minor of the beam profile, originating from beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=SigmaMajorOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=sigma_major,
                source_units="px",
                destination_units="px",
                attributes={
                    "note": "Sigma major of the beam profile, originating from beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=ThetaOutPath,
                minimum_dimensionality=3,
                data_type="float32",
                default_value=theta,
                source_units="radians",
                destination_units="radians",
                attributes={
                    "note": "Theta of the beam profile, originating from beam_analysis post-translation processing script."
                },
            )
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)
        logger.info(f"Completed beam information for {input_file}")

    except subprocess.CalledProcessError as e:
        logger.info("beam information failed with error:")
        logger.info(e)
        logger.error(f"Error during beam information step: {e}")
