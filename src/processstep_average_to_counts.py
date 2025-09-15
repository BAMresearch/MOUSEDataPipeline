from pathlib import Path
import subprocess

import h5py
import numpy as np
from YMD_class import YMD, extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from pint import UnitRegistry
from utilities import get_float_from_h5

# Initialize the unit registry
ureg = UnitRegistry()

doc = """
This processing step converts the averaged frame data (in instrument/detector00/data) and its
uncertainties to counts by multiplying with the number of frames. 
It also adjusts the count_time accordingly.
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"metadata_updater cannot run in {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Updates some metadata fields in the process with a new version of the metadata from the logbook and project/sample information.
    """

    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    # get the logbook entry for this measurement
    n_frames = get_float_from_h5(input_file, '/entry1/instrument/detector00/averaged_number_of_frames', logger)

    if n_frames == 0.0:
        logger.warning(f"could not find number of averaged frames in {input_file}, skipping update")
        return

    convert_paths = [
        '/entry1/instrument/detector00/data',
        '/entry1/instrument/detector00/data_uncertainties_poisson',
        '/entry1/instrument/detector00/data_uncertainties_sem',
        '/entry1/instrument/detector00/count_time',
    ]

    try:
        logger.info(f"Starting average_to_counts updater for {input_file}")

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for path in convert_paths:
                # let's hope I can do this inplace
                h5_out[path][...] *= n_frames
                oldnote = h5_out[path].attrs.get('note', '')
                if isinstance(oldnote, bytes):
                    oldnote = oldnote.decode('utf-8')

                # update the notes to reflect the situation
                h5_out[path].attrs['note'] = f"Converted from averaged data to counts by multiplying with {n_frames} frames. \n Original note: {oldnote}"

        logger.info(f"Completed average_to_counts for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e)
