from fileinput import filename
from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from checkers import len_files_in_path, processing_possible
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from utilities import reduce_extra_image_dimensions, prepare_eiger_image, label_main_feature
from skimage.measure import regionprops
import numpy as np
import h5py
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from utilities import get_pint_quantity_from_h5
from HDF5Translator.utils import Q_


def diameter_from_distance(distance_m: float, reference_distance_m: float = 2, reference_diameter_in_px: float = 600) -> float:
    """
    Returns the diameter of the beam mask around the beam center, by scaling the reference diameter at reference distance to the new distance. 
    This ensures that the solid angle coverage is similar for the different distances so that a consistent transmission calculation can be done.
    """
    return float(reference_diameter_in_px * distance_m / reference_distance_m)


def generate_mask(image_shape, center, radius, logger: logging.Logger) -> np.ndarray:
    """
    Generates a circular mask with given center and radius. This should not exceed the image boundaries.
    0 outside the mask, 1 inside the mask.
    """
    row, col = np.ogrid[:image_shape[0], :image_shape[1]]
    dist_from_center = np.sqrt((row - center[0])**2 + (col - center[1])**2)
    mask = dist_from_center <= radius
    return mask


# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the beam mask determination can run. We need the translated file.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beam mask determination not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the beam mask determination processing step. This version determines the appropriate beam mask
    based on the distance between sample and detector, so that the solid angle coverage is similar. 
    This means that for the longest distance (s-d approx 2m), we use the largest possible circular mask 
    around the direct beam, that still fits on the detector. For shorter distances, we reduce the mask diameter
    proportionally.
    """
    # get beam center information from the weighted center of mass. 
    COMPath = "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"
    # we also need the detector position along the beam: 
    detZPath = "/entry1/instrument/detector00/transformations/det_x"
    # and the sample offset along the same axis: 
    sampleOffsetZPath = "/entry1/sample/transformations/sample_x"
    BeamMaskPath = "/entry1/processing/direct_beam_profile/beam_analysis/BeamMask"

    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:

        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'

        logger.info(f"Starting beam mask determination for {input_file}")

        # center of mass in px:
        with h5py.File(input_file, "r") as h5_in:
            # Read necessary information (this is just a placeholder, adapt as needed)
            COM = h5_in[COMPath][()]
            # image dimensions:
            imShape = h5_in['/entry1/instrument/detector00/data'][()].squeeze().shape
        # detector distance:
        detZ = get_pint_quantity_from_h5(input_file, detZPath, logger)
        # sample offset:
        sampleOffsetZ = get_pint_quantity_from_h5(input_file, sampleOffsetZPath, logger)
        distance = (detZ - sampleOffsetZ).to('m')

        assert np.isfinite(distance.magnitude) and distance.magnitude > 0.0, f"invalid sample-detector distance {distance} from {detZPath} and {sampleOffsetZPath} in {input_file}"
        # determine the appropriate diameter:
        diameter = diameter_from_distance(distance.magnitude, reference_distance_m=2.0, reference_diameter_in_px=600)
        mask = generate_mask(imShape, COM, diameter / 2, logger)
        # print(f'image {imShape=}, mask shape {mask.shape=}, {COM=}, {distance=}, {diameter=}')

        # Write out the beam center:
        TElements = []  # we want to add two elements, so I make a list
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=BeamMaskPath,
                minimum_dimensionality=3,
                data_type="float32",
                default_value=mask,
                source_units="dimensionless",
                destination_units="dimensionless",
                attributes={
                    "note": "Mask used for the beam intensity determination, originating from beam_analysis post-translation processing script."
                },
            )
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)
        logger.info(f"Completed beam mask determination for {input_file}")

    except subprocess.CalledProcessError as e:
        logger.info("beam mask determination failed with error:")
        logger.info(e)
        logger.error(f"Error during beam mask determination step: {e}")
