from fileinput import filename
from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from checkers import len_files_in_path, processing_possible
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from utilities import get_float_from_h5, reduce_extra_image_dimensions, prepare_eiger_image, label_main_feature
from skimage.measure import regionprops
import numpy as np
import h5py
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from utilities import get_pint_quantity_from_h5
from HDF5Translator.utils import Q_


# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


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
