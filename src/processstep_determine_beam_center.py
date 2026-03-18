import logging
import subprocess
from pathlib import Path

import h5py
from HDF5Translator.translator import process_translation_element
from HDF5Translator.translator_elements import TranslationElement
from skimage.measure import regionprops

from defaults_carrier import DefaultsCarrier
from logbook_support import LogbookReaderLike
from utilities import get_weighted_centroid_compat, label_main_feature, prepare_eiger_image
from YMD_class import extract_metadata_from_path

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False


def can_run(
    dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger
) -> bool:
    """
    Checks if the beam center determination should run. We need the translated file.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}.nxs"
    if not step_2_file.is_file():
        logger.info(f"Beam center determination not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: LogbookReaderLike | None, logger: logging.Logger):
    """
    Executes the beam center determination processing step.
    """

    COMOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"
    xOutPath = "/entry1/instrument/detector00/transformations/det_y"
    zOutPath = "/entry1/instrument/detector00/transformations/det_z"
    BeamDatapath = "/entry1/processing/direct_beam_profile/data"
    # BeamDurationPath = (
    #     "/entry1/processing/direct_beam_profile/frame_time"
    # )
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:
        input_file = dir_path / f"MOUSE_{ymd}_{batch}_{repetition}.nxs"

        logger.info(f"Starting beam analysis for {input_file}")

        # step 1: read image
        with h5py.File(input_file, "r") as h5_in:
            # Read necessary information (this is just a placeholder, adapt as needed)
            imageData = h5_in[BeamDatapath][()]
            # mean because count_time is the frame time minus the readout time.
            # recordingTime = h5_in[BeamDurationPath][()]
        maskedTwoDImage = prepare_eiger_image(imageData, logger)
        # create labeled mask:
        logger.info(f"Looking for main feature in beam center determination for {input_file}")
        labels = label_main_feature(maskedTwoDImage, logger)
        # step 4: calculate region properties
        properties = regionprops(labels, maskedTwoDImage)  # calculate region properties
        weighted_center_of_mass = get_weighted_centroid_compat(properties[0])  # center of mass (weighted)

        # Write out the beam center:
        TElements = []  # we want to add multiple elements, so I make a list
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=COMOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=weighted_center_of_mass,
                source_units="px",
                destination_units="px",
                attributes={
                    "note": "Intensity weighted center of mass, determined by the beam center processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=xOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=weighted_center_of_mass[1],
                source_units="eigerpixels",
                destination_units="m",
                attributes={
                    "note": "Determined by the beam center processing script.",
                    "depends_on": "./det_z",
                    "offset": "[0.0,0.0,0.0]",
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "vector": "[1.0,0.0,0.0]",
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=zOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=weighted_center_of_mass[0],
                source_units="eigerpixels",
                destination_units="m",
                attributes={
                    "note": "Determined by the beam center processing script.",
                    "depends_on": "./det_x",
                    "offset": "[0.0,0.0,0.0]",
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "vector": "[0.0,1.0,0.0]",
                },
            ),
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)
        logger.info(f"Completed beam center determination for {input_file}")

    except subprocess.CalledProcessError as e:
        logger.info("beam center determination failed with error:")
        logger.info(e)
        logger.error(f"Error during beam center determination step: {e}")
