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

This example determines a beam center, transmission and flux from a beamstopless measurement.
The path can be specified on the command line, meaning the same operation can be used on the 
direct beam measurement as well as the sample beam measurement. The ROI size can be specified. 

This is an operation which is normally done in the MOUSE procedure
requires scikit-image
"""

import argparse
from ctypes import Union
import logging
from pathlib import Path
from typing import Tuple
import hdf5plugin  # loaded BEFORE h5py
import h5py
import numpy as np
from skimage.measure import regionprops
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
This script determines reflectivity around the expected location of the specular reflection.

The expected height (based on calibration - SD and alignment - angle) is
calculated and skimage.regionprops used on a region of shape (4*ROI_SIZE, 4*ROI_SIZE).
The 
"""


def reflectionAnalysis(imageData: np.ndarray, ROI_SIZE: int,
                       direct_beam: tuple,
                       incident_angle: float,
                       distance: float,
                       pixel_size_y: float) -> (tuple, float):
    """
    Perform beam analysis on the given image data, returning the beam center and flux.
    """
    # Step 1: reducing the dimensionality of the imageData by averaging until we have a 2D array:
    while imageData.ndim > 2:
        imageData = np.mean(imageData, axis=0)

    # Step 2: get rid of masked or pegged pixels on an Eiger detector
    labeled_foreground = (np.logical_and(imageData >= 0, imageData <= 1e9)).astype(int)
    maskedTwoDImage = imageData * labeled_foreground  # apply mask
    
    # Step 3: ROI where the reflected beam is expected
    # vertical is first entry in direct_beam tuple

    def reflection_pixel(incident_angle, distance, direct_beam):
        height = 2*np.tan(np.deg2rad(incident_angle)) * distance
        pixel_reflected = direct_beam[0] - height/pixel_size_y
        return pixel_reflected

    # use a range of angles - alignment may not be perfect
    angle_error = 0.01
    pixel_reflected = np.array([reflection_pixel(a, distance, direct_beam) for a in [incident_angle + angle_error, incident_angle - angle_error]])

    reflection_slice0 = slice(np.max(int(pixel_reflected.min() - 1*ROI_SIZE), 0), np.minimum(
        int(pixel_reflected.max() + 1*ROI_SIZE), maskedTwoDImage.shape[0]))
    reflection_slice1 = slice(np.max(int(direct_beam[1] - 1*ROI_SIZE), 0), np.minimum(
        int(direct_beam[1] + 1*ROI_SIZE), maskedTwoDImage.shape[1]))
    reflection_slice = slice(reflection_slice0, reflection_slice1)

    beam_slice0 = slice(np.max(int(direct_beam[0] - 1*ROI_SIZE), 0), np.minimum(
        int(direct_beam[0] + 1*ROI_SIZE), maskedTwoDImage.shape[0]))
    beam_slice1 = slice(np.max(int(direct_beam[1] - 1*ROI_SIZE), 0), np.minimum(
        int(direct_beam[1] + 1*ROI_SIZE), maskedTwoDImage.shape[1]))

    mask_beam = np.ones(imageData.shape)
    mask_beam[beam_slice0, beam_slice1] = 0
    maskedTwoDImage *= mask_beam
    label_reflection = np.zeros(imageData.shape)
    #
    label_reflection[reflection_slice0, reflection_slice1] = 1
    maskedTwoDImage *= label_reflection
    
    threshold_value = np.maximum(
        1, 1e-6 * maskedTwoDImage.max()
    )  # filters.threshold_otsu(maskedTwoDImage) # ignore zero pixels
    labeled_peak = (maskedTwoDImage > threshold_value).astype(int)  # label peak
    properties = regionprops(labeled_peak, imageData)  # calculate region properties
    center_of_mass = None
    if len(properties) > 0:
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
    else:
        weighted_center_of_mass = None
        ITotal_region = 0 
    # for your info:
    logging.debug(f"{center_of_mass=}")
    logging.debug(f"{ITotal_region=} counts")

    return weighted_center_of_mass, ITotal_region

def reflectionAnalysisBeamSubtracted(imageData: np.ndarray, ROI_SIZE: int,
                                     direct_beam_data: np.ndarray,
                                     incident_angle: float,
                                     distance: float,
                                     pixel_size_y: float) -> (tuple, float):

    def reduce_and_mask(data):
        while data.ndim > 2:
            data = np.mean(data, axis=0)

        # Step 2: get rid of masked or pegged pixels on an Eiger detector
        labeled_foreground = (np.logical_and(data >= 0, data <= 1e9)).astype(int)
        maskedTwoDImage = data * labeled_foreground  # apply mask
        return maskedTwoDImage

    imageData = reduce_and_mask(imageData)
    direct_beam_data = reduce_and_mask(direct_beam_data)

    # subtract direct beam from image
    image_sub = imageData - direct_beam_data

    threshold_value = np.maximum(
        1, 1e-6 * image_sub.max()
    )  # filters.threshold_otsu(image_sub) # ignore zero pixels
    labeled_peak = (image_sub > threshold_value).astype(int)  # label peak
    properties = regionprops(labeled_peak, imageData)  # calculate region properties
    center_of_mass = None
    if len(properties) > 0:
        center_of_mass = properties[0].centroid  # center of mass (unweighted by intensity)
        weighted_center_of_mass = properties[
            0
        ].weighted_centroid  # center of mass (weighted)
        # determine the total intensity in the region of interest, this will be later divided by measuremet time to get the flux
        ITotal_region = np.sum(
            image_sub[
                np.maximum(int(weighted_center_of_mass[0] - ROI_SIZE), 0) : np.minimum(
                    int(weighted_center_of_mass[0] + ROI_SIZE), image_sub.shape[0]
                ),
                np.maximum(int(weighted_center_of_mass[1] - ROI_SIZE), 0) : np.minimum(
                    int(weighted_center_of_mass[1] + ROI_SIZE), image_sub.shape[1]
                ),
            ]
        )
    else:
        weighted_center_of_mass = None
        ITotal_region = 0 
    # for your info:
    logging.debug(f"{center_of_mass=}")
    logging.debug(f"{ITotal_region=} counts")

    return weighted_center_of_mass, ITotal_region




def apparent_angle(beam_com, refl_com, distance, pixelsize):
    height_px = beam_com - refl_com
    height_m = height_px * pixelsize
    apparent_angle = 0.5*np.rad2deg(np.arctan(height_m/distance))
    return apparent_angle


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
    FootprintPath = "/entry1/processing/specular_reflection/footprint"
    ReflectionOutPath = "/entry1/processing/specular_reflection/reflectivity"
    ReflectionFluxOutPath = "/entry1/processing/specular_reflection/flux"
    ReflectionPositionOutPath = "/entry1/processing/specular_reflection/centerOfMass"
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
    DirectFluxPath = "/entry1/sample/beam/flux"
    DetectorDistancePath = "/entry1/instrument/detector00/transformations/det_x"
    PixelSizeYPath = "/entry1/instrument/detector00/y_pixel_size"
    

    # reading from the main HDF5 file
    with h5py.File(filename, "r") as h5_in:
        # Read necessary information (this is just a placeholder, adapt as needed)
        imageData = h5_in[DataPath][()]
        recordingTime = h5_in[DurationPath][()]
        beamrecordingTime = h5_in[BeamDurationPath][()]
        beamcenter = h5_in[COMPath][()]
        beamflux = h5_in[DirectFluxPath][()]
        incident_angle = h5_in["/entry1/sample/transformations/meridional_angle"][()]
        wavelength = h5_in[BeamWavelengthPath][()]
        distance = h5_in[DetectorDistancePath][()].item()
        pixel_size_y = h5_in[PixelSizeYPath][()].item()
        transmission = h5_in[TransmissionPath][()].item()
        direct_beam_data = h5_in[BeamDataPath][()]
        footprint = h5_in[FootprintPath][()]
        

    # Now you can do operations, such as determining a beam center and flux. For that, we need to
    # do a few steps...

    
    center_of_mass, ITotal_region = reflectionAnalysis(imageData, ROI_SIZE, beamcenter, incident_angle, distance, pixel_size_y)
    if ITotal_region is not None:
        logging.info(
            f"Beam center: {center_of_mass}, Flux: {ITotal_region / recordingTime} counts/s."
        )
    else:
        logging.info(
            f"No reflected beam in image."
        )

    if ITotal_region is not None and center_of_mass is not None:
            angle_apparent = apparent_angle(beamcenter[0], center_of_mass[0], distance, pixel_size_y)
    else:
        angle_apparent = np.abs(incident_angle)
        center_of_mass = (0,0)
        center_of_mass, ITotal_region = reflectionAnalysisBeamSubtracted(imageData, ROI_SIZE, direct_beam_data, incident_angle, distance, pixel_size_y)

        
        
    qz = 4*np.pi/wavelength*np.sin(np.deg2rad(incident_angle))
    qz_apparent = 4*np.pi/wavelength*np.sin(np.deg2rad(angle_apparent))
    
    

    # Now we start the write-back to the HDF5 file, using the TranslationElement class
    # This class lets you configure exactly what the output should look like in the HDF5 file.
    TElements = []  # we want to add two elements, so I make a list

    
    
    
    TElements += [
        TranslationElement(
            # source is none since we're storing derived data
            destination=incidentAngleApparentPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=angle_apparent,
            source_units="deg",
            destination_units="deg",
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=incidentAnglePath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=incident_angle,
            source_units="deg",
            destination_units="deg",
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=qzOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=qz[0],
            source_units="1/nm",
            destination_units="1/nm",
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=qz_apparentOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=qz_apparent[0],
            source_units="1/nm",
            destination_units="1/nm",
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=ReflectionPositionOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=center_of_mass,
            source_units="px",
            destination_units="px",
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=ReflectionFluxOutPath,
            default_value=ITotal_region / recordingTime,
            data_type="float",
            destination_units="counts/s",
            minimum_dimensionality=1,
            attributes={
                "note": "Determined by the reflectionanalysis post-translation processing script."
            },
        ),
    ]
    xOutPath = None
    zOutPath = None 

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
                    "note": "Determined by the reflectionanalysis post-translation processing script.",
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
                    "note": "Determined by the reflectionanalysis post-translation processing script.",
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
        directBeamFlux = h5_in.get(DirectFluxPath, default=None)
        sampleBeamFlux = h5_in.get(ReflectionFluxOutPath, default=None)
        directBeamFlux = directBeamFlux[()] if directBeamFlux is not None else None
        sampleBeamFlux = sampleBeamFlux[()] if sampleBeamFlux is not None else None

    sampleBeamFlux = ITotal_region / recordingTime

    if directBeamFlux is not None and sampleBeamFlux is not None:
        reflection = sampleBeamFlux / (directBeamFlux * footprint)
        logging.info(f"Adding reflectivity to the file: {reflection}")
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=ReflectionOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=reflection,
                destination_units="",
                attributes={
                    "note": "Determined by the reflectionanalysis post-translation processing script."
                },
            )            
            
        ]

    # writing the resulting metadata back to the main HDF5 file
    with h5py.File(filename, "r+") as h5_out:
        for element in TElements:  # iterate over the two elements and write them back
            process_translation_element(None, h5_out, element)

        # set NX data attributes for easy plotting
        h5_out[reflectionGroupPath].attrs.update({"NX_class": "NXdata", 'signal': 'reflectivity', 'axes': 'qz'})


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
