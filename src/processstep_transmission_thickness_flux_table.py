


from pathlib import Path
import subprocess

import h5py
import numpy as np
from YMD_class import extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
import csv


doc = """
This processing step updates the metadata with the estimated thickness from the 
X-ray absorption and the X-ray absorption coefficient calculated from the composition. 
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = False

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beamanalysis not possible for {dir_path}, file missing at: {step_2_file}")
        return False

    return True

def get_float_from_h5(filename: Path, HDFPath: str, logger: logging.Logger) -> float:
    """
    Returns the value from the HDF5 file at HDFPath.
    """
    try: 
        with h5py.File(filename, 'r') as h5f:
            val = h5f[HDFPath][()]
        if isinstance(val, list) or isinstance(val, np.ndarray): 
            val = np.mean(val)
    except Exception as e:
        logger.warning(f'could not read absorption coefficient from {filename} with error {e}')
        return 0.0
    if not isinstance(val, np.floating):
        logger.warning(f'absorption coefficient not found in file {filename}')
        return 0.0
    return val

def get_str_from_h5(filename: Path, HDFPath: str, logger: logging.Logger) -> float:
    """
    Returns the value from the HDF5 file at HDFPath.
    """
    try: 
        with h5py.File(filename, 'r') as h5f:
            val = h5f[HDFPath][()].decode('utf-8')
    except Exception as e:
        logger.warning(f'could not read value {HDFPath} from {filename} with error {e}')
        return ''
    return val


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the processing step.
    This step should read the following information from the HDF5 files (right before stacking), and write the following information in a separate csv table file:
    - filename
    - ymd
    - batch
    - repetition
    - flux: "/entry1/sample/flux"
    - transmission: "/entry1/sample/transmission"
    - absorption_total: "/entry1/sample/absorption_total"  
    - absorption_by_sample: "/entry1/sample/absorption_by_sample"
    - absorption_by_bg: "/entry1/sample/absorption_by_bg"
    - absorption_coefficient: "/entry1/sample/overall_mu"
    - composition: "/entry1/sample/composition"
    - absorption_derived_thickness: "/entry1/sample/absorptionDerivedThickness"
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    output_file = dir_path.parent / f'MOUSE_flux_thickness_table.csv'

    try:
        logger.info(f"Starting thickness_from_absorption step for {input_file}")
      # Gather the needed data
        data = {
            'filename': input_file.name,
            'ymd': ymd,
            'batch': batch,
            'repetition': repetition,
            'flux': get_float_from_h5(input_file, "/entry1/sample/beam/flux", logger),
            'transmission': get_float_from_h5(input_file, "/entry1/sample/transmission", logger),
            'absorption_total': get_float_from_h5(input_file, "/entry1/sample/absorption_total", logger),
            'absorption_by_sample': get_float_from_h5(input_file, "/entry1/sample/absorption_by_sample", logger),
            'absorption_by_bg': get_float_from_h5(input_file, "/entry1/sample/absorption_by_bg", logger),
            'absorption_coefficient': get_float_from_h5(input_file, "/entry1/sample/overall_mu", logger),
            'composition': get_str_from_h5(input_file, "/entry1/sample/composition", logger),
            'absorption_derived_thickness': get_float_from_h5(input_file, "/entry1/sample/absorptionDerivedThickness", logger),
        }

        # Check if the file exists to decide on writing headers
        file_exists = output_file.is_file()

        # Write to CSV with append mode
        with output_file.open('a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if file doesn't exist
            writer.writerow(data)

        logger.info(f"Completed table generation step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info("table generation step failed with error:")
        logger.info(e)
        logger.error(f"Error during table generation step: {e}")