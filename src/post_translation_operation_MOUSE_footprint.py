#!/usr/bin/env python
# coding: utf-8

"""
Post-Translation HDF5 Processor

This script performs post-translation steps on HDF5 files, including reading information,
performing calculations (e.g. for determining beam centers, transmission factors and other 
derived information), and writes the result back into the HDF5 structure of the original file.

Usage:
    python post_translation_processor.py --input measurement.h5 [--auxilary_files file2.h5 ...] [-v]

Replace the calculation and file read/write logic according to your specific requirements.

This script determines the footprint based on the samplelength, direct beam profile,
incident angle and measured transmission, assuming a Gaussian beam profile. 

This is an operation which is normally done for GIXS and XRR
requires scipy
"""

import argparse
from ctypes import Union
import logging
from pathlib import Path
from typing import Tuple
import hdf5plugin  # loaded BEFORE h5py
import h5py
import numpy as np
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit
from HDF5Translator.utils.data_utils import sanitize_attribute
from HDF5Translator.utils.validators import (
    file_check_extension,
    file_exists_and_is_file,
)
from HDF5Translator.utils.configure_logging import configure_logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from HDF5Translator.utils.data_utils import getFromKeyVals

description = """
This script determines the footprint based on the samplelength, direct beam profile,
incident angle and measured transmission, assuming a Gaussian beam profile. 
"""

def beam_center_from_transmission(transmission, incident_angle, length, beam_sigma):
    return length * np.sin(np.deg2rad(incident_angle)) - erfinv(2*(1-transmission) - 1)*beam_sigma

def pitchgi_footprint(x, x0, length, ampl, beam_center, beam_sigma):
    return ampl * (0.5*(erf((length * np.abs(np.sin((x-x0)*np.pi/180.)) # upper edge
                             - beam_center)/beam_sigma)
                        + 1
                        )
                   - 0.5*(erf((length * (np.sin((-x-x0)*np.pi/180.)) # lower edge
                               - beam_center)/beam_sigma)
                          + 1
                          )
                   )




def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)



def profileAnalysis(imageData: np.ndarray, ROI_SIZE: int,
                    direct_beam: tuple,
                    pixel_size_y: float) -> float:
    """
    Perform beam analysis on the given image data, returning the beam center and flux.
    """
    # Step 1: reducing the dimensionality of the imageData by averaging until we have a 2D array:
    while imageData.ndim > 2:
        imageData = np.mean(imageData, axis=0)

    # Step 2: get rid of masked or pegged pixels on an Eiger detector
    labeled_foreground = (np.logical_and(imageData >= 0, imageData <= 1e9)).astype(int)
    maskedTwoDImage = imageData * labeled_foreground  # apply mask
    
    # Step 3: ROI around the direct beam
    # vertical is first entry in direct_beam tuple

    beam_slice0 = slice(np.max(int(direct_beam[0] - 1*ROI_SIZE), 0), np.minimum(
        int(direct_beam[0] + 1*ROI_SIZE), maskedTwoDImage.shape[0]))
    beam_slice1 = slice(np.max(int(direct_beam[1] - 1*ROI_SIZE), 0), np.minimum(
        int(direct_beam[1] + 1*ROI_SIZE), maskedTwoDImage.shape[1]))

    maskedTwoDImage = maskedTwoDImage[beam_slice0, beam_slice1]

    # Step 4: fit a Gaussian profile

    profile = np.mean(maskedTwoDImage, axis = 1)
    x = np.arange(profile.size)
    popt, _ = curve_fit(gaussian, x, profile)
    amplitude, mean, std = popt

    # convert to mm
    std *= pixel_size_y
    
    # for your info:
    logging.debug(f"{mean=}")
    logging.debug(f"{std=} mm")

    return std

# If you are adjusting the template for your needs, you probably only need to touch the main function:
def main(
    filename: Path,
    auxilary_files: list[Path] | None = None,
    keyvals: dict | None = None,
):
    """
    We do a three-step process here:
      1. read from the main HDF5 file (and optionally the auxilary files),
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
    logging.info(
        f"Processing image in file {filename} with ROI size of {ROI_SIZE} pixels."
    )

    # Define the paths in the HDF5 file where the data is stored and where the results should be written
    incidentAnglePath = "/entry1/processing/incident_angle"
    incidentAngleApparentPath = "/entry1/processing/incident_angle_apparent"
    reflectionGroupPath = "/entry1/processing/specular_reflection/"
    qzOutPath = "/entry1/processing/specular_reflection/qz"
    qz_apparentOutPath = "/entry1/processing/specular_reflection/qz_apparent"
    FootprintOutPath = "/entry1/processing/specular_reflection/footprint"
    OffsetOutPath = "/entry1/processing/specular_reflection/beam_offset"
    
    DataPath = "/entry1/instrument/detector00/data"
    BeamDataPath = "/entry1/processing/direct_beam_profile/data"  # 2 frames
    BeamDurationPath = (
            "/entry1/instrument/detector00/count_time"
            #"/entry1/instrument/detector00/detectorSpecific/frame_count_time"
        )
    BeamWavelengthPath = (
        "/entry1/sample/beam/incident_wavelength"
        )
    DurationPath = BeamDurationPath#(
        #"/entry1/processing/measurement_detector_output/instrument/detector/frame_time"
        #)
    COMPath = "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"
    TransmissionPath = "/entry1/sample/transmission"
    PixelSizeYPath = "/entry1/instrument/detector00/y_pixel_size"
    SampleLengthPath = "/entry1/sample/length"
    

    # reading from the main HDF5 file
    with h5py.File(filename, "r") as h5_in:
        # Read necessary information (this is just a placeholder, adapt as needed)
        imageData = h5_in[DataPath][()]
        recordingTime = h5_in[DurationPath][()]
        beamrecordingTime = h5_in[BeamDurationPath][()]
        beamcenter = h5_in[COMPath][()]
        incident_angle = h5_in["/entry1/sample/transformations/meridional_angle"][()]
        wavelength = h5_in[BeamWavelengthPath][()]
        pixel_size_y = h5_in[PixelSizeYPath][()].item()
        transmission = h5_in[TransmissionPath][()].item()
        direct_beam_data = h5_in[BeamDataPath][()]
        sample_length = h5_in[SampleLengthPath][()]
        

    logging.info(f"{sample_length=}, {incident_angle=}")
    # Now you can do operations, such as determining a beam center and flux. For that, we need to
    # do a few steps...

    footprint = 1 - 2*transmission
    if footprint < 0:
        sigma_beam = profileAnalysis(direct_beam_data, ROI_SIZE, beamcenter, pixel_size_y*1e3)
        logging.info(f"beam sigma determined: {sigma_beam} mm")

        beam_offset = beam_center_from_transmission(transmission, incident_angle, sample_length, sigma_beam)
        if beam_offset is not None:
            logging.info(
                f"Beam offset: {beam_offset} mm, std: {sigma_beam} mm."
            )
        else:
            logging.info(
                f"Beam width and offset could not be determined."
            )
        footprint = pitchgi_footprint(incident_angle, 0, sample_length, 1, beam_offset, sigma_beam)
    else:
        beam_offset = 0



    # Now we start the write-back to the HDF5 file, using the TranslationElement class
    # This class lets you configure exactly what the output should look like in the HDF5 file.
    TElements = []  # we want to add two elements, so I make a list

    
    
    
    TElements += [
        TranslationElement(
            # source is none since we're storing derived data
            destination=FootprintOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=footprint,
            source_units="",
            destination_units="",
            attributes={
                "note": "Determined by the footprint post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=OffsetOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=beam_offset,
            source_units="mm",
            destination_units="mm",
            attributes={
                "note": "Determined by the footprint post-translation processing script."
            },
        ),
    ]

    # writing the resulting metadata back to the main HDF5 file
    with h5py.File(filename, "r+") as h5_out:
        for element in TElements:  # iterate over the two elements and write them back
            process_translation_element(None, h5_out, element)



    logging.info("Post-translation processing complete.")


### The code below probably does not need changing for use of the tremplate. ###


def validate_file(file_path: str | Path) -> Path:
    """
    Validates that the file exists and has a valid extension.

    Args:
        file_path (str): Path to the file to validate.

    Returns:
        Path: Path object of the file.
    """
    file_path = Path(file_path)
    file_exists_and_is_file(file_path)
    file_check_extension(file_path, [".h5", ".hdf5", ".nxs", ".H5", ".HDF5", ".NXS"])
    return file_path


# for handling the optional keyval arguments for feeding additional parameters to your operation
class KeyValueAction(argparse.Action):
    """
    Custom action for argparse to parse key-value pairs from the command line.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        keyvals = {}
        for item in values:
            # Split on the first equals sign
            key, value = item.split("=", 1)
            keyvals[key.strip()] = value.strip()
        setattr(namespace, self.dest, keyvals)


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
        "--auxilary_files",
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
    if args.auxilary_files:
        for auxilary_file in args.auxilary_files:
            logging.info(f"using auxilary file: {auxilary_file}")

    main(args.filename, args.auxilary_files, args.keyvals)
