from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
# from checkers import len_files_in_path, processing_possible
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader  # type: ignore
import logging
from utilities import get_float_from_h5, prepare_eiger_image, label_main_feature
from skimage import measure
from skimage.measure import regionprops
import numpy as np
import h5py
from HDF5Translator.translator_elements import TranslationElement  # type: ignore
from HDF5Translator.translator import process_translation_element  # type: ignore
# from utilities import get_pint_quantity_from_h5
# from HDF5Translator.utils import Q_  # type: ignore
from typing import Optional, Union


# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def dynamic_beam_analysis(imageData: np.ndarray, coverage: float = 0.997, beam_coverage_mask: Optional[np.ndarray] = None) -> Union[tuple, float, np.ndarray]:
    """
    This method is used to calculate the beam mask and properties based on either a provided mask or
    by determining a tight beam mask based on a coverage target. Its intended use is to get a rough
    estimate for the scattering probability, by comparing the beam intensity under the tight mask
    with the total scattered image intensity on the beam_through_sample image.
    """

    def _beam_coverage_mask_from_regionprops(
            reg: measure._regionprops.RegionProperties,
            shape,
            coverage: float
            ) -> Union[np.ndarray, np.ndarray]:
        """
        Build a full-image boolean mask of the k·σ ellipse defined by the
        region's intensity-weighted centroid and covariance, where k gives
        the requested 2-D Gaussian coverage. - disclaimer: this method is mainly GPT code...
        """
        # 1) center and weighted moments
        cy, cx = reg.weighted_centroid
        mu_c = reg.weighted_moments_central
        m00 = reg.weighted_moments[0, 0]
        if m00 <= 0:
            return np.zeros(shape, dtype=bool)

        # 2) covariance (row, col)
        var_r = mu_c[0, 2] / m00
        var_c = mu_c[2, 0] / m00
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
            k_lo: float = 0.5,
            k_hi: float = 5.0,
            steps: int = 8
            ) -> float:
        """
            Real peaks deviate from perfect Gaussians. This bisection nudges k
            so the integrated fraction over your peak matches the target: - also GPT code
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

    # if we don't have a mask yet, we need to determine one (for direct_beam only. sample_beam should use the direct beam mask)
    # Step 1: get rid of masked or pegged pixels on an Eiger detector
    maskedTwoDImage = prepare_eiger_image(imageData, logging.getLogger())
    sigma_minor, sigma_major, theta = None, None, None
    if beam_coverage_mask is not None:
        assert beam_coverage_mask.shape == maskedTwoDImage.shape, "Provided beam_coverage_mask must have the same shape as imageData"
        beam_coverage_mask = beam_coverage_mask.astype(int)
    else:

        labels = label_main_feature(maskedTwoDImage, logging.getLogger())
        # step 4: calculate region properties
        properties = regionprops(labels, maskedTwoDImage)  # calculate initial region properties

        # GPT addition:
        coverage_target = coverage
        beam_coverage_mask, md2, sigma_minor, sigma_major, theta = _beam_coverage_mask_from_regionprops(
            properties[0],
            maskedTwoDImage.shape,
            coverage_target
            )
        # Keep the ellipse inside the original label to avoid bleeding into neighbors
        # beam_coverage_mask &= (labels.astype(bool))
        # refine for the actual peak not a gaussian peak:
        k = refine_k_for_exact_coverage(md2, (labels > 0), maskedTwoDImage, coverage_target)
        beam_coverage_mask = (md2 <= k*k) & (labels > 0)
        kept_intensity = float(maskedTwoDImage[beam_coverage_mask].sum())
        achieved_coverage = kept_intensity / properties[0].intensity_image.sum()
        print(f"Refined k={k:.3f} to achieve coverage {achieved_coverage:.4f} ({coverage_target=})")
        beam_coverage_mask = beam_coverage_mask.astype(int)

    properties = regionprops(beam_coverage_mask, maskedTwoDImage)  # calculate region properties
    # continue normally if beam found
    # center_of_mass = properties[0].centroid  # center of mass (unweighted by intensity)
    weighted_center_of_mass = properties[0].weighted_centroid  # center of mass (weighted)
    # get the intensity in the region of interest
    ITotal_region = float(properties[0].intensity_image.sum())
    ITotal_overall = float(maskedTwoDImage.sum())

    return weighted_center_of_mass, ITotal_region, ITotal_overall, beam_coverage_mask, sigma_minor, sigma_major, theta


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the beam flux and transmissions determination can run. We need the translated file.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beam flux and transmissions determination not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    After the beam center and beam masks have been determined, we can now calculate the beam flux and transmissions.
    We will calculate two transmission factors: the image transmission, which is the ratio of total intensity in the
    beam and beam-through-sample images, and the beam transmission, which is the ratio of the intensity for those
    images under the beam mask only.
    A correction factor is then calculated as the ratio of the two transmissions, which can be used to correct
    the transmission factor for the scattering contribution in further steps.
    """
    readPaths = {
        "DarkcurrentPath": "/entry1/instrument/detector00/darkcurrent",
        "DirectBeamDataPath": "/entry1/processing/direct_beam_profile/data",
        "DirectBeamDurationPath": "/entry1/processing/direct_beam_profile/frame_time",
        "SampleBeamDataPath": "/entry1/processing/sample_beam_profile/data",
        "SampleBeamDurationPath": "/entry1/processing/sample_beam_profile/frame_time",
        "BeamMaskPath": "/entry1/processing/direct_beam_profile/beam_analysis/BeamMask",
        # Define the paths in the HDF5 file where the data is stored and where the results should be written
        "TransmissionOutPath": "/entry1/sample/transmission",
        "ImageTransmissionOutPath": "/entry1/sample/transmission_image",
        "TransmissionCorrectionFactorOutPath": "/entry1/sample/transmission_correction_factor",
        "SampleFluxOutPath": "/entry1/processing/sample_beam_profile/beam_analysis/flux",
        "DirectFluxOutPath": "/entry1/sample/beam/flux",
        "SampleFluxOverImagePath": "/entry1/processing/sample_beam_profile/beam_analysis/FluxOverImage",
        "DirectFluxOverImagePath": "/entry1/processing/direct_beam_profile/beam_analysis/FluxOverImage",
        "ScatteringProbabilityEstimatePath": "/entry1/sample/scattering_probability_estimate",
        "TightBeamMaskPath": "/entry1/processing/direct_beam_profile/beam_analysis/tight_beam_mask",
    }

    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:
        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'

        logger.info(f"Starting beam flux and transmissions determination for {input_file}")

        # center of mass in px:
        with h5py.File(input_file, "r") as h5_in:
            # Read necessary information (this is just a placeholder, adapt as needed)
            DirectBeamData = prepare_eiger_image(h5_in[readPaths['DirectBeamDataPath']][()], logger)
            DirectBeamDuration = get_float_from_h5(input_file, readPaths['DirectBeamDurationPath'], logger)
            SampleBeamData = prepare_eiger_image(h5_in[readPaths['SampleBeamDataPath']][()], logger)
            SampleBeamDuration = get_float_from_h5(input_file, readPaths['SampleBeamDurationPath'], logger)
            BeamMask = prepare_eiger_image(h5_in[readPaths['BeamMaskPath']][()], logger)
            Darkcurrent = get_float_from_h5(input_file, readPaths['DarkcurrentPath'], logger)

        # compute the needed values:
        DBFluxImage = DirectBeamData / DirectBeamDuration - Darkcurrent
        SBFluxImage = SampleBeamData / SampleBeamDuration - Darkcurrent

        DirectFluxOverImage = np.nansum(DBFluxImage)
        SampleFluxOverImage = np.nansum(SBFluxImage)
        DirectFlux = np.nansum(DBFluxImage * BeamMask)
        SampleFlux = np.nansum(SBFluxImage * BeamMask)
        # print(f'{repetition=}, {DirectFluxOverImage=:0.02f}, {SampleFluxOverImage=:0.02f}, {DirectFlux=:0.02f}, {SampleFlux=:0.02f}, {Darkcurrent=:0.02f}')
        ImageTransmission = SampleFluxOverImage / DirectFluxOverImage
        Transmission = SampleFlux / DirectFlux
        TransmissionCorrectionFactor = ImageTransmission / Transmission

        # now we calculate the estimate for the multiple scattering based on a tight beam mask on the direct beam image:
        _, _, _, tightBeamMask, _, _, _ = dynamic_beam_analysis(DirectBeamData, coverage=0.997, beam_coverage_mask=None)
        # and determine the fluxes in the sample beam image under that mask:
        _, sample_tight_beam_flux, sample_overall_flux, _, _, _, _ = dynamic_beam_analysis(SampleBeamData, coverage=0.997, beam_coverage_mask=tightBeamMask)
        scattering_probability_estimate = (sample_overall_flux - sample_tight_beam_flux) / sample_overall_flux

        # print(f'{repetition=}, {ImageTransmission=:0.08f}, {Transmission=:0.08f}, {TransmissionCorrectionFactor=:0.08f}')
        # write the findings back to the file:
        TElements = []  # we want to add two elements, so I make a list
        TElements += [
            TranslationElement(
                destination=readPaths['DirectFluxOverImagePath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=DirectFluxOverImage,
                attributes={
                    "note": "(Darkcurrent-compensated) beam flux over the entire image, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="counts/second",
                destination_units="counts/second",
            ),
            TranslationElement(
                destination=readPaths['SampleFluxOverImagePath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=SampleFluxOverImage,
                attributes={
                    "note": "(Darkcurrent-compensated) beam flux over the entire image with sample in beam, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="counts/second",
                destination_units="counts/second",
            ),
            TranslationElement(
                destination=readPaths['DirectFluxOutPath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=DirectFlux,
                attributes={
                    "note": "Beam flux under the beam mask, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="counts/second",
                destination_units="counts/second",
            ),
            TranslationElement(
                destination=readPaths['SampleFluxOutPath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=SampleFlux,
                attributes={
                    "note": "Beam flux under the beam mask with sample in beam, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="counts/second",
                destination_units="counts/second",
            ),
            TranslationElement(
                destination=readPaths['ImageTransmissionOutPath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=ImageTransmission,
                attributes={
                    "note": "Image transmission factor (ratio of sample to direct beam flux over entire image), determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="dimensionless",
                destination_units="dimensionless",
            ),
            TranslationElement(
                destination=readPaths['TransmissionOutPath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=Transmission,
                attributes={
                    "note": "Beam transmission factor (ratio of sample to direct beam flux under beam mask), determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="dimensionless",
                destination_units="dimensionless",
            ),
            TranslationElement(
                destination=readPaths['TransmissionCorrectionFactorOutPath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=TransmissionCorrectionFactor,
                attributes={
                    "note": "Correction factor applied to the image transmission, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="dimensionless",
                destination_units="dimensionless",
            ),
            TranslationElement(
                destination=readPaths['ScatteringProbabilityEstimatePath'],
                data_type="float",
                minimum_dimensionality=1,
                default_value=scattering_probability_estimate,
                attributes={
                    "note": "Estimate for the scattering probability based on a tight beam mask on the direct beam image, determined by the beam flux and transmissions post-translation processing script",
                },
                source_units="dimensionless",
                destination_units="dimensionless",
            ),
            TranslationElement(
                destination=readPaths['TightBeamMaskPath'],
                data_type="float32",
                minimum_dimensionality=3,
                default_value=tightBeamMask,
                attributes={
                    "note": "Tight beam mask determined from the direct beam image, used for the scattering probability estimate, originating from beam_flux_and_transmissions post-translation processing script."
                },
                source_units="dimensionless",
                destination_units="dimensionless",
            )
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)
        logger.info(f"Completed beam flux and transmissions determination for {input_file}")

    except subprocess.CalledProcessError as e:
        logger.info("beam flux and transmissions determination failed with error:")
        logger.info(e)
        logger.error(f"Error during beam flux and transmissions determination step: {e}")
