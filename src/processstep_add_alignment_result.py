from pathlib import Path
import subprocess

import h5py
from YMD_class import YMD, extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element

doc = """
This processing step finds the alignment result file for this measurement and converts meridional to incident angle
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run.
    """
    if "scan_" in dir_path.stem:
        ymd, batch, repetition = extract_metadata_from_path(dir_path.parent.parent)
    else:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_1_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs'
    if not step_1_file.is_file():
        logger.info(f"Alignment info addition not possible for {dir_path}, file missing at: {step_1_file}")
        return False
    return True

def get_pitch(filename: Path, logger: logging.Logger) -> float:
    """
    Read the .nxs file in aligned configuration return its pitch
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            pitchgi = h5f['/saxs/Saxslab/pitchgi'][()]
    except Exception as e:
        logger.error(f"Error reading pitch from file: {e}")
        pitchgi = 0
    return pitchgi.astype("float")

def findentry(ymd:YMD, batch:int, logbook_reader: Logbook2MouseReader):
    # print(f'searching for {ymd.YMD} and {batch}, type {type(ymd.YMD)} and {type(batch)}')
    batch = int(batch)
    for entry in logbook_reader.entries:
        # print(f'checking {entry.ymd} and {entry.batchnum}, type {type(entry.ymd)} and {type(entry.batchnum)}')
        if entry.ymd == ymd.YMD and entry.batchnum == batch:
            return entry
    return None

from pathlib import Path
import logging
from datetime import datetime

def find_aligned_file(defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, measurement_ymd: YMD, batch: int, logger: logging.Logger) -> Path:
    """
    Finds the appropriate alignment result file based on measurement ymd and alignment_batch parameter
    
    :param defaults: An instance of DefaultsCarrier containing default paths.
    :param logbook_reader: An instance of the Logbook2MouseReader. 
    :param measurement_ymd: `ymd` string of the measurement in format YYYYMMDD.
    :param batch: Batch number to match.
    :param logger: Logger object for logging purposes.
    :return: Path to the appropriate mask file.
    """
    entry = findentry(measurement_ymd, batch, logbook_reader)
    # specify batch in the logbook
    alignment_batch = int(float(entry.additional_parameters.get("alignment_batch", 1)))
    # repetition is one - after the alignment scans which are in the *_0 subdir
    data_dir = defaults.saxs_dir / "data" / measurement_ymd.get_year() / str(measurement_ymd) / f"{str(measurement_ymd)}_{alignment_batch}_1"
    aligned_files = list(data_dir.glob("*.nxs"))

    if len(aligned_files) > 0:
        aligned_file = aligned_files[0]
        logger.info(f"Selected alignment result file: {aligned_file}")
        return aligned_file
    else:
        logger.warning(f"No suitable alignment found for ymd {measurement_ymd} batch {batch}.")
    

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    if "scan_" in dir_path.stem:
        ymd, batch, repetition = extract_metadata_from_path(dir_path.parent.parent)
    else:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)

    try:
        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs'
        logger.info(f"Starting alignment correction for {input_file}")

        aligned_file = find_aligned_file(defaults, logbook_reader, ymd, batch, logger)
        # print(f'* * * * * * * * Found mask file: {mask_file} for configuration {input_file}')
        if aligned_file is None:
            logger.error(f"No suitable alignment data found for repetition {ymd}_{batch}_{repetition}.")
            return
        horizontal_pitch = get_pitch(aligned_file, logger)
        print(horizontal_pitch)
        # let's add the aligment data from that alignment result file to the input file, using HDF5Translator elements:
        TElements = [
            TranslationElement(
                source="/entry1/sample/transformations/meridional_angle",
                destination="/entry1/sample/transformations/meridional_angle",
                source_units="deg",
                destination_units="deg",
                transformation=f'lambda x: {horizontal_pitch} - float(x[0])',
                attributes={
                    "note": f"Added from the alignment result file {aligned_file.as_posix()} by the processstep_add_alignment_result.",
                },
            ),
        ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_in:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(h5_in, h5_in, element)

        logger.info(f"Completed translator step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info(f"Processstep processstep_add_alignment_result failed with stderr:")
        logger.info(e)
        logger.error(f"Error during translator subprocess: {e}")
