#!/usr/bin/env python
# coding: utf-8

"""
Post-Translation HDF5 Processor

This script performs post-translation steps on HDF5 files, including reading information,
performing calculations (e.g. for determining beam centers, transmission factors and other 
derived information), and writes the result back into the HDF5 structure of the original file.

Usage:
    python post_translation_processor.py --input measurement.h5 [--auxiliary_files file2.h5 ...] [-v]

Replace the calculation and file read/write logic according to your specific requirements.

This example determines a beam center, transmission and flux from a beamstopless measurement.
The path can be specified on the command line, meaning the same operation can be used on the 
direct beam measurement as well as the sample beam measurement. The ROI size can be specified. 

This is an operation which is normally done in the MOUSE procedure
requires scikit-image
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Union
import hdf5plugin  # loaded BEFORE h5py
import h5py
import numpy as np
from skimage.measure import regionprops
from HDF5Translator.utils.data_utils import sanitize_attribute
from HDF5Translator.utils.validators import (
    validate_file
)
from HDF5Translator.utils.argparse_utils import KeyValueAction
from HDF5Translator.utils.configure_logging import configure_logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from HDF5Translator.utils.data_utils import getFromKeyVals

description = """
This script is an example on how to perform post-translation operations on HDF5 files.
Ths example includes all the necessary steps, such as including reading information,
performing calculations (e.g. for determining beam centers, transmission factors and other
derived information), and writes the result back into the HDF5 structure of the original file.

The example includes universal command-line arguments for specifying the input file,
auxiliary files, and verbosity level. It includes validators and a logging engine. There is also 
the option of supplying key-value pairs for additional parameters to your operation.

You can replace the calculation and file read/write logic according to your specific requirements.
"""

def hdf5_get_image(filename: Path, h5imagepath: str = "entry/data/data") -> np.ndarray:
    with h5py.File(filename, "r") as h5f:
        image = h5f[h5imagepath][()]
    return image

def reduce_extra_image_dimensions(image:np.ndarray, method=np.mean)->np.ndarray:
    assert method in [np.mean, np.sum], "method must be either np.mean or np.sum function handles"
    while image.ndim > 2:
        image = method(image, axis=0)
    return image

def beam_analysis(imageData: np.ndarray, ROI_SIZE: int) -> Union[tuple, float]:
    """
    Perform beam analysis on the given image data, returning the beam center and flux.
    """

    # Step 1: get rid of masked or pegged pixels on an Eiger detector
    labeled_foreground = (np.logical_and(imageData >= 0, imageData <= 1e9)).astype(int)
    maskedTwoDImage = imageData * labeled_foreground  # apply mask
    threshold_value = np.maximum(
        1, 0.0001 * maskedTwoDImage.max()
    )  # filters.threshold_otsu(maskedTwoDImage) # ignore zero pixels
    labeled_peak = (maskedTwoDImage > threshold_value).astype(int)  # label peak
    properties = regionprops(labeled_peak, imageData)  # calculate region properties
    if len(properties) == 0:  # no beam found
        return (0,0), 0
    # continue normally if beam found
    center_of_mass = properties[0].centroid  # center of mass (unweighted by intensity)
    weighted_center_of_mass = properties[
        0
    ].weighted_centroid  # center of mass (weighted)
    # determine the total intensity in the region of interest, this will be later divided by measuremet time to get the flux
    ITotal_region = np.sum(
        maskedTwoDImage[
            np.maximum(int(weighted_center_of_mass[0] - ROI_SIZE), 0) : np.minimum(
                int(weighted_center_of_mass[0] + ROI_SIZE), maskedTwoDImage.shape[0]
            ),
            np.maximum(int(weighted_center_of_mass[1] - ROI_SIZE), 0) : np.minimum(
                int(weighted_center_of_mass[1] + ROI_SIZE), maskedTwoDImage.shape[1]
            ),
        ]
    )
    # for your info:
    logging.debug(f"{center_of_mass=}")
    logging.debug(f"{ITotal_region=} counts")

    return center_of_mass, ITotal_region


# If you are adjusting the template for your needs, you probably only need to touch the main function:
def main(
    filename: Path,
    auxiliary_files: list[Path] | None = None,
    keyvals: dict | None = None,
):
    """
    We do a three-step process here:
      1. read from the main HDF5 file (and optionally the auxiliary files),
      2. perform an operation, in this example determining the beam center and flux,
      3. and write back to the file

    In this template, we have the example of determining the beam parameters (center location, flux) from
    the detector data of a beamstopless measurement, and writing it back to the HDF5 file. The example
    also shows how you can add command-line inputs to your process as well.
    """
    # Process input parameters:
    # Define the size of the region of interest (ROI) for beam center determination (in +/- pixels from center)
    ROI_SIZE = getFromKeyVals(
        "roi_size", keyvals, 25
    )  # Size of the region of interest (ROI) for beam center determination. your beam center should be at least this far from the edge
    imageType = getFromKeyVals(
        "image_type", keyvals, "direct_beam"
    )  # can be either direct_beam or sample_beam. This sets the paths and the output location
    logging.info(
        f"Processing {imageType} image in file {filename} with ROI size of {ROI_SIZE} pixels."
    )

    # Define the paths in the HDF5 file where the data is stored and where the results should be written
    TransmissionOutPath = "/entry1/sample/transmission"
    SampleFluxOutPath = "/entry1/processing/sample_beam_profile/beam_analysis/flux"
    DirectFluxOutPath = "/entry1/sample/beam/flux"
    if imageType == "direct_beam":
        BeamDatapath = "/entry1/processing/direct_beam_profile/data/data_000001"
        BeamDurationPath = (
            "/entry1/processing/direct_beam_profile/instrument/detector/count_time"
        )
        COMOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"
        xOutPath = "/entry1/instrument/detector00/transformations/det_y"
        zOutPath = "/entry1/instrument/detector00/transformations/det_z"
        FluxOutPath = DirectFluxOutPath
    elif imageType == "sample_beam":
        BeamDatapath = "/entry1/processing/sample_beam_profile/data/data_000001"
        BeamDurationPath = (
            "/entry1/processing/sample_beam_profile/instrument/detector/count_time"
        )
        COMOutPath = "/entry1/processing/sample_beam_profile/beam_analysis/centerOfMass"
        xOutPath = None  # no need to store these as we get the beam center from the direct beam
        zOutPath = None
        FluxOutPath = SampleFluxOutPath

    else:
        logging.error(
            f"Unknown image type: {imageType}. Please specify either 'direct_beam' or 'sample_beam'."
        )
        return

    # print(f'{BeamDatapath=}, {BeamDurationPath=}, {COMOutPath=}, {xOutPath=}, {zOutPath=}, {FluxOutPath=}')

    # reading from the main HDF5 file
    with h5py.File(filename, "r") as h5_in:
        # Read necessary information (this is just a placeholder, adapt as needed)
        imageData = h5_in[BeamDatapath][()]
        # mean because count_time is the frame time minus the readout time. 
        imageData = reduce_extra_image_dimensions(imageData, method=np.mean)
        recordingTime = h5_in[BeamDurationPath][()]

    # Now you can do operations, such as determining a beam center and flux. For that, we need to
    # do a few steps...
    center_of_mass, ITotal_region = beam_analysis(imageData, ROI_SIZE)
    logging.info(
        f"Beam center: {center_of_mass}, Flux: {ITotal_region / recordingTime} counts/s."
    )
    # Now we start the write-back to the HDF5 file, using the TranslationElement class
    # This class lets you configure exactly what the output should look like in the HDF5 file.
    TElements = []  # we want to add two elements, so I make a list
    TElements += [
        TranslationElement(
            # source is none since we're storing derived data
            destination=COMOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=center_of_mass,
            source_units="px",
            destination_units="px",
            attributes={
                "note": "Determined by the beam_analysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=FluxOutPath,
            default_value=ITotal_region / recordingTime,
            data_type="float",
            destination_units="counts/s",
            minimum_dimensionality=1,
            attributes={
                "note": "Determined by the beam_analysis post-translation processing script."
            },
        ),
    ]

    if xOutPath is not None and zOutPath is not None:
        logging.info("Direct beam center found, storing in detector transformations.")
        # if we have the direct beam, we can also store the beam center in the detector transformations
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=xOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=center_of_mass[1],
                source_units="eigerpixels",
                destination_units="m",
                attributes={
                    "note": "Determined by the beam_analysis post-translation processing script.",
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
                default_value=center_of_mass[0],
                source_units="eigerpixels",
                destination_units="m",
                attributes={
                    "note": "Determined by the beam_analysis post-translation processing script.",
                    "depends_on": "./det_x",
                    "offset": "[0.0,0.0,0.0]",
                    "offset_units": "m",
                    "transformation_type": "translation",
                    "vector": "[0.0,1.0,0.0]",
                },
            ),
        ]

    # find out if we have enough information to calcuate the transmission factor:
    with h5py.File(filename, "r") as h5_in:
        directBeamFlux = h5_in.get(DirectFluxOutPath, default=None)
        sampleBeamFlux = h5_in.get(SampleFluxOutPath, default=None)
        directBeamFlux = directBeamFlux[()] if directBeamFlux is not None else None
        sampleBeamFlux = sampleBeamFlux[()] if sampleBeamFlux is not None else None
    if imageType == "direct_beam":
        directBeamFlux = ITotal_region / recordingTime
    elif imageType == "sample_beam":
        sampleBeamFlux = ITotal_region / recordingTime

    if directBeamFlux is not None and sampleBeamFlux is not None:
        transmission = sampleBeamFlux / directBeamFlux
        logging.info(f"Adding transmission factor to the file: {transmission}")
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=TransmissionOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=transmission,
                destination_units="",
                attributes={
                    "note": "Determined by the beam_analysis post-translation processing script."
                },
            )
        ]

    # writing the resulting metadata back to the main HDF5 file
    with h5py.File(filename, "r+") as h5_out:
        for element in TElements:  # iterate over the two elements and write them back
            process_translation_element(None, h5_out, element)

    logging.info("Post-translation processing complete.")


### The code below probably does not need changing for use of the tremplate. ###



def setup_argparser():
    """
    Sets up command line argument parser using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=validate_file,
        required=True,
        help="Input measurement HDF5 file.",
    )
    parser.add_argument(
        "-a",
        "--auxiliary_files",
        type=validate_file,
        nargs="*",
        help="Optional additional HDF5 files needed for processing. (read-only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity to INFO level.",
    )
    parser.add_argument(
        "-vv",
        "--very_verbose",
        action="store_true",
        help="Increase output verbosity to DEBUG level.",
    )
    parser.add_argument(
        "-l",
        "--logging",
        action="store_true",
        help="Write log out to a timestamped file.",
    )
    parser.add_argument(
        "-k",
        "--keyvals",
        nargs="+",
        action=KeyValueAction,
        help="Optional key-value pairs (key=value)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Entry point for the script. Parses command line arguments and calls the main function.
    """
    args = setup_argparser()
    configure_logging(
        args.verbose,
        args.very_verbose,
        log_to_file=args.logging,
        log_file_prepend="PostTranslationProcessor_",
    )

    logging.info(f"Processing input file: {args.filename}")
    if args.auxiliary_files:
        for auxiliary_file in args.auxiliary_files:
            logging.info(f"using auxiliary file: {auxiliary_file}")

    main(args.filename, args.auxiliary_files, args.keyvals)
