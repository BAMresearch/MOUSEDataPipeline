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
from typing import Tuple, Union, Optional
import hdf5plugin  # loaded BEFORE h5py
import h5py
import numpy as np
from skimage.measure import regionprops
from skimage import measure, morphology # for new beam analysis
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

# def hdf5_get_image(filename: Path, h5imagepath: str = "entry/data/data") -> np.ndarray:
#     with h5py.File(filename, "r") as h5f:
#         image = h5f[h5imagepath][()]
#     return image


def reduce_extra_image_dimensions(image:np.ndarray, method=np.mean)->np.ndarray:
    assert method in [np.mean, np.sum, np.any, np.all], "method must be either np.mean or np.sum function handles"
    while image.ndim > 2:
        image = method(image, axis=0)
    return image


def new_beam_analysis(imageData: np.ndarray, coverage: float = 0.90, ellipse_mask:Optional[np.ndarray] = None) -> Union[tuple, float, np.ndarray]:
    
    def _ellipse_mask_from_regionprops(
            reg: measure._regionprops.RegionProperties,
            shape,
            coverage: float
            ) -> Union[np.ndarray, np.ndarray]:
        """
        Build a full-image boolean mask of the k·σ ellipse defined by the
        region's intensity-weighted centroid and covariance, where k gives
        the requested 2-D Gaussian coverage. - GPT code...
        """
        # 1) center and weighted moments
        cy, cx = reg.weighted_centroid
        mu_c = reg.weighted_moments_central
        m00  = reg.weighted_moments[0, 0]
        if m00 <= 0:
            return np.zeros(shape, dtype=bool)

        # 2) covariance (row, col)
        var_r  = mu_c[0, 2] / m00
        var_c  = mu_c[2, 0] / m00
        cov_rc = mu_c[1, 1] / m00
        cov = np.array([[var_r, cov_rc],
                        [cov_rc, var_c]], dtype=float)
        cov = (cov + cov.T) / 2.0  # symmetrize
        cov_inv = np.linalg.inv(cov + 1e-12 * np.eye(2))

        # 3) coverage -> radius k
        coverage = float(np.clip(coverage, 1e-6, 1 - 1e-9))
        k = float(np.sqrt(-2.0 * np.log(1.0 - coverage)))

        # 4) Mahalanobis distance mask
        rr, cc = np.indices(shape)
        dr = rr - cy
        dc = cc - cx
        md2 = (
            cov_inv[0, 0]*dr*dr +
            2.0*cov_inv[0, 1]*dr*dc +
            cov_inv[1, 1]*dc*dc
            )
        
        # --- Peak widths (σ) and orientation ---
        evals, evecs = np.linalg.eigh(cov)  # eigenvalues ascending
        evals = np.clip(evals, 0.0, None)      # no negative variances
        sigma_minor, sigma_major = np.sqrt(evals[0]), np.sqrt(evals[1])

        # Orientation of major axis (CCW from row-axis)
        v_major = evecs[:, 1]
        theta = float(np.arctan2(v_major[0], v_major[1]))

        return md2 <= (k**2), md2, sigma_minor, sigma_major, theta

    def refine_k_for_exact_coverage(
            md2, 
            base_mask, 
            img, 
            target, 
            k_lo:float=0.5, 
            k_hi:float=5.0, 
            steps:int=8
            ) -> float:
        """
            Real peaks deviate from perfect Gaussians. This 10-line bisection nudges k
            so the integrated fraction over your peak matches the target: - GPT code
        """
        total = float(img[base_mask].sum())
        for _ in range(steps):
            k_mid = 0.5*(k_lo + k_hi)
            frac = float(img[(md2 <= k_mid*k_mid) & base_mask].sum()) / total
            if frac < target:
                k_lo = k_mid
            else:
                k_hi = k_mid
        return 0.5*(k_lo + k_hi)


    # Now you can do operations, such as determining a beam center and flux. For that, we need to
    # do a few steps...

    # if we don't have a mask yet, we need to determine one (for direct_beam only. sample_beam should use the direct beam mask)
    # Step 1: get rid of masked or pegged pixels on an Eiger detector
    labeled_foreground = (np.logical_and(imageData >= 0, imageData <= 2e7)).astype(int)
    maskedTwoDImage = imageData * labeled_foreground  # apply mask
    sigma_minor, sigma_major, theta = None, None, None
    if ellipse_mask is not None:
        assert ellipse_mask.shape == imageData.shape, "Provided ellipse_mask must have the same shape as imageData"
        ellipse_mask = ellipse_mask.astype(int)
    else:
        threshold_value = np.maximum(
            1, 1*maskedTwoDImage.mean() # 0.0001 * maskedTwoDImage.max()
        )  # filters.threshold_otsu(maskedTwoDImage) # ignore zero pixels
        # print(f'{threshold_value=}')

        # Step 1: binary mask of "candidate bright regions"
        mask = maskedTwoDImage > threshold_value

        # Step 2: label connected components
        # labels = measure.label(mask, connectivity=2)
        labels, num = measure.label(
            morphology.convex_hull_image(  # we expect the beam to be convex
                morphology.remove_small_holes(  # with moly we may see small dead pixels in the beam
                    morphology.remove_small_objects(  # we don't care about isolated spikes
                        mask,
                        min_size=20
                        ),
                    area_threshold=20
                    ),
                ),
            connectivity=1,
            return_num=True
            )

        # Step 3: ensure we only have the main feature
        if num == 0:
            raise ValueError("No beam found in the image.")
        if num > 1:
            print("Multiple beams found in the image, selecting the largest one.")
            # find the largest component and keep only that one
            largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # skip background label 0
            labels = (labels == largest_label).astype(int)
        # assert we only have one labeled region now
        assert np.unique(labels).size == 2, "More than one labeled region found."

        # step 4: calculate region properties
        properties = regionprops(labels, imageData)  # calculate region properties

        # GPT addition:
        # --- NEW: shrink the region to desired intensity coverage (e.g., 95%) ---
        coverage_target = coverage
        ellipse_mask, md2, sigma_minor, sigma_major, theta = _ellipse_mask_from_regionprops(
            properties[0],
            imageData.shape,
            coverage_target
            )
        # Keep the ellipse inside the original label to avoid bleeding into neighbors
        # ellipse_mask &= (labels.astype(bool))
        # refine for the actual peak not a gaussian peak: 
        k = refine_k_for_exact_coverage(md2, (labels > 0), maskedTwoDImage, coverage_target)
        ellipse_mask = (md2 <= k*k) & (labels > 0)
        kept_intensity = float(maskedTwoDImage[ellipse_mask].sum())
        achieved_coverage = kept_intensity / properties[0].intensity_image.sum()
        print(f"Refined k={k:.3f} to achieve coverage {achieved_coverage:.4f} ({coverage_target=})")
        ellipse_mask = ellipse_mask.astype(int)

    properties = regionprops(ellipse_mask, imageData)  # calculate region properties
    # continue normally if beam found
    # center_of_mass = properties[0].centroid  # center of mass (unweighted by intensity)
    weighted_center_of_mass = properties[0].weighted_centroid  # center of mass (weighted)
    # get the intensity in the region of interest
    ITotal_region = float(properties[0].intensity_image.sum())
    ITotal_overall = float(maskedTwoDImage.sum())

    return weighted_center_of_mass, ITotal_region, ITotal_overall, ellipse_mask, sigma_minor, sigma_major, theta


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
    # coverageTarget = getFromKeyVals(
    #     "roi_size", keyvals, 25
    # )  # Size of the region of interest (ROI) for beam center determination. your beam center should be at least this far from the edge
    imageType = getFromKeyVals(
        "image_type", keyvals, "direct_beam"
    )  # can be either direct_beam or sample_beam. This sets the paths and the output location
    logging.info(
        f"Processing {imageType} image in file {filename}"
    )

    # Define the paths in the HDF5 file where the data is stored and where the results should be written
    TransmissionOutPath = "/entry1/sample/transmission"
    ImageTransmissionOutPath = "/entry1/sample/transmission_image"
    TransmissionCorrectionFactorOutPath = "/entry1/sample/transmission_correction_factor"
    SigmaMinorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_minor"
    SigmaMajorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_major"
    ThetaOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/theta"

    SampleFluxOutPath = "/entry1/processing/sample_beam_profile/beam_analysis/flux"
    DirectFluxOutPath = "/entry1/sample/beam/flux"
    SampleFluxOverImagePath = "/entry1/processing/direct_beam_profile/beam_analysis/FluxOverImage"
    DirectFluxOverImagePath = "/entry1/processing/sample_beam_profile/beam_analysis/FluxOverImage"
    BeamMaskPath = "/entry1/processing/direct_beam_profile/beam_analysis/BeamMask"
    if imageType == "direct_beam":
        BeamDatapath = "/entry1/processing/direct_beam_profile/data"
        BeamDurationPath = (
            "/entry1/processing/direct_beam_profile/frame_time"
        )
        COMOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"
        xOutPath = "/entry1/instrument/detector00/transformations/det_y"
        zOutPath = "/entry1/instrument/detector00/transformations/det_z"
        FluxOutPath = DirectFluxOutPath
        FluxOverImagePath = DirectFluxOverImagePath
    elif imageType == "sample_beam":
        BeamDatapath = "/entry1/processing/sample_beam_profile/data"
        BeamDurationPath = (
            "/entry1/processing/sample_beam_profile/frame_time"
        )
        COMOutPath = "/entry1/processing/sample_beam_profile/beam_analysis/centerOfMass"
        xOutPath = None  # no need to store these as we get the beam center from the direct beam
        zOutPath = None
        FluxOutPath = SampleFluxOutPath
        FluxOverImagePath = SampleFluxOverImagePath

    else:
        logging.error(
            f"Unknown image type: {imageType}. Please specify either 'direct_beam' or 'sample_beam'."
        )
        return

    # reading from the main HDF5 file
    with h5py.File(filename, "r") as h5_in:
        # Read necessary information (this is just a placeholder, adapt as needed)
        imageData = h5_in[BeamDatapath][()]
        # mean because count_time is the frame time minus the readout time. 
        recordingTime = h5_in[BeamDurationPath][()]
        # read the beam mask if it exists (for sample beam analysis), otherwise None
        ellipse_mask = h5_in.get(BeamMaskPath, default=None)
        # reduce dimensions if needed
        if ellipse_mask is not None:
            ellipse_mask = ellipse_mask[()]
            ellipse_mask = reduce_extra_image_dimensions(ellipse_mask, method=np.any)
        else:
            ellipse_mask = None

    if imageType == "sample_beam":
        assert ellipse_mask is not None, "For sample_beam analysis, the beam mask must be provided from the direct_beam analysis."

    imageData = reduce_extra_image_dimensions(imageData, method=np.mean)

    # Now you can do operations, such as determining a beam center and flux. For that, we need to
    # do a few steps...
    # center_of_mass, ITotal_region = beam_analysis(imageData, ROI_SIZE)
    center_of_mass, ITotal_region, ITotal_overall, ellipse_mask, sigma_minor, sigma_major, theta = new_beam_analysis(
        imageData,
        coverage=0.90,
        ellipse_mask=ellipse_mask
        )
    logging.info(
        f"Beam center: {center_of_mass}, Flux: {ITotal_region / recordingTime} counts/s."
    )
    # Now we start the write-back to the HDF5 file, using the TranslationElement class
    # This class lets you configure exactly what the output should look like in the HDF5 file.
    TElements = []  # we want to add two elements, so I make a list
    TElements += [
        TranslationElement(
            # source is none since we're storing derived data
            destination=BeamMaskPath,
            minimum_dimensionality=3,
            data_type="float32",
            default_value=ellipse_mask,
            source_units="px",
            destination_units="px",
            attributes={
                "note": "Mask used for the beam intensity determination, originating from beam_analysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=COMOutPath,
            minimum_dimensionality=1,
            data_type="float32",
            default_value=center_of_mass,
            source_units="px",
            destination_units="px",
            attributes={
                "note": "Intensity weighted center of mass, determined by the beam_analysis post-translation processing script."
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
                "note": "Flux over the beam only, determined by the beam_analysis post-translation processing script."
            },
        ),
        TranslationElement(
            # source is none since we're storing derived data
            destination=FluxOverImagePath,
            default_value=ITotal_overall / recordingTime,
            data_type="float",
            destination_units="counts/s",
            minimum_dimensionality=1,
            attributes={
                "note": "Flux over the total image, determined by the beam_analysis post-translation processing script."
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
        directImageFlux = h5_in.get(DirectFluxOverImagePath, default=None)
        sampleImageFlux = h5_in.get(SampleFluxOverImagePath, default=None)
        directImageFlux = directImageFlux[()] if directImageFlux is not None else None
        sampleImageFlux = sampleImageFlux[()] if sampleImageFlux is not None else None

    if imageType == "direct_beam":
        directBeamFlux = ITotal_region / recordingTime
        directImageFlux = ITotal_overall / recordingTime
    elif imageType == "sample_beam":
        sampleBeamFlux = ITotal_region / recordingTime
        sampleImageFlux = ITotal_overall / recordingTime

    if directBeamFlux is not None and sampleBeamFlux is not None:
        transmission = sampleBeamFlux / directBeamFlux
        transmission_image = sampleImageFlux / directImageFlux
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
                    "note": "Beam-based transmission factor by the beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=ImageTransmissionOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=transmission_image,
                destination_units="",
                attributes={
                    "note": "Image-based transmission factor, determined by the beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                destination=TransmissionCorrectionFactorOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=transmission_image/transmission,
                destination_units="",
                attributes={
                    "note": "Correction factor to multiply with beam transmission to get approximate true transmission, determined by the beam_analysis post-translation processing script, overwritten later by the value from the measurement with the closest detector position."
                },
            )
        ]

    if sigma_minor is not None and sigma_major is not None and theta is not None:
        logging.info("Adding beam profile parameters to the file.")
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination=SigmaMinorOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=sigma_minor,
                destination_units="px",
                attributes={
                    "note": "Minor sigma of the elliptical Gaussian fit to the beam profile, determined by the beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=SigmaMajorOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=sigma_major,
                destination_units="px",
                attributes={
                    "note": "Major sigma of the elliptical Gaussian fit to the beam profile, determined by the beam_analysis post-translation processing script."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination=ThetaOutPath,
                minimum_dimensionality=1,
                data_type="float32",
                default_value=theta,
                destination_units="radians",
                attributes={
                    "note": "Angle of the major axis of the elliptical Gaussian fit to the beam profile (CCW from row axis), determined by the beam_analysis post-translation processing script."
                },
            ),
        ]

    # writing the resulting metadata back to the main HDF5 file
    with h5py.File(filename, "r+") as h5_out:
        for element in TElements:  # iterate over the two elements and write them back
            process_translation_element(None, h5_out, element)

    logging.info("Post-translation processing complete.")


# The code below probably does not need changing for use of the tremplate.
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
