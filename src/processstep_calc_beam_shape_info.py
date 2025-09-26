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
from utilities import get_pint_quantity_from_h5, get_float_from_h5
from HDF5Translator.utils import Q_
from skimage import measure
from typing import Optional, Union

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def new_beam_analysis(imageData: np.ndarray, coverage: float = 0.997, ellipse_mask:Optional[np.ndarray] = None) -> Union[tuple, float, np.ndarray]:
    
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
            Real peaks deviate from perfect Gaussians. This bisection nudges k
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
    maskedTwoDImage = imageData
    assert maskedTwoDImage.ndim == 2, "imageData must be a 2D array. Did you run prepare_eiger_image on it?"

    sigma_minor, sigma_major, theta = None, None, None
    if ellipse_mask is not None:
        assert ellipse_mask.shape == maskedTwoDImage.shape, "Provided ellipse_mask must have the same shape as imageData"
        ellipse_mask = ellipse_mask.astype(int)
    else:

        labels = label_main_feature(maskedTwoDImage, logging.getLogger())
        # step 4: calculate region properties
        properties = regionprops(labels, maskedTwoDImage)  # calculate initial region properties

        # GPT addition:
        coverage_target = coverage
        ellipse_mask, md2, sigma_minor, sigma_major, theta = _ellipse_mask_from_regionprops(
            properties[0],
            maskedTwoDImage.shape,
            coverage_target
            )
        # Keep the ellipse inside the original label to avoid bleeding into neighbors
        # ellipse_mask &= (labels.astype(bool))
        # refine for the actual peak not a gaussian peak: 
        k = refine_k_for_exact_coverage(md2, (labels > 0), maskedTwoDImage, coverage_target)
        ellipse_mask = (md2 <= k*k) & (labels > 0)
        kept_intensity = float(maskedTwoDImage[ellipse_mask].sum())
        achieved_coverage = kept_intensity / properties[0].intensity_image.sum()
        # print(f"Refined k={k:.3f} to achieve coverage {achieved_coverage:.4f} ({coverage_target=})")
        ellipse_mask = ellipse_mask.astype(int)

    properties = regionprops(ellipse_mask, maskedTwoDImage)  # calculate region properties
    # continue normally if beam found
    # center_of_mass = properties[0].centroid  # center of mass (unweighted by intensity)
    weighted_center_of_mass = properties[0].weighted_centroid  # center of mass (weighted)
    # get the intensity in the region of interest
    ITotal_region = float(properties[0].intensity_image.sum())
    ITotal_overall = float(maskedTwoDImage.sum())

    return weighted_center_of_mass, ITotal_region, ITotal_overall, ellipse_mask, sigma_minor, sigma_major, theta


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
    DirectBeamDurationPath = (
        "/entry1/processing/direct_beam_profile/frame_time"
    )
    SigmaMinorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_minor"
    SigmaMajorOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/sigma_major"
    ThetaOutPath = "/entry1/processing/direct_beam_profile/beam_analysis/theta"

    BeamMaskPath = "/entry1/processing/direct_beam_profile/beam_analysis/BeamMask"

    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:

        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'

        logger.info(f"Starting beam info determination for {input_file}")

        with h5py.File(input_file, "r") as h5_in:
            # Read necessary information (this is just a placeholder, adapt as needed)
            DirectBeamData = prepare_eiger_image(h5_in[DirectBeamDatapath][()], logger)
            DirectBeamDuration = get_float_from_h5(input_file, DirectBeamDurationPath, logger)

        # compute the needed values:
        weighted_center_of_mass, ITotal_region, ITotal_overall, ellipse_mask, sigma_minor, sigma_major, theta = new_beam_analysis(
            DirectBeamData, coverage=0.997, ellipse_mask=None
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
