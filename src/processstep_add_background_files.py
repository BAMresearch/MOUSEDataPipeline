from pathlib import Path
import subprocess
from typing import Tuple, Union

import h5py
from YMD_class import YMD, extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging

from processstep_add_mask_file import get_configuration

doc = """
This processing step adds the stacked background files (whether they exist or not) for this measurement. 
Should be run after the metadata update step.
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    if not step_2_file.is_file():
        logger.info(f"Background files adding not possible for {dir_path}, file missing at: {step_2_file}")
        return False
    return True

def get_background_identifiers(filename: Path, logger: logging.Logger) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Read the file and return the background and dispersant background identifiers. If not specified, return None.
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            bgid = h5f['/entry1/processing_required_metadata/background_identifier'][()].decode('utf-8')
            dbgid = h5f['/entry1/processing_required_metadata/dispersant_background_identifier'][()].decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading background identifiers from file: {e}")
        bgid = None
        dbgid = None

    if bgid == 'None': bgid = None
    if dbgid == 'None': dbgid = None

    return bgid, dbgid


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the translator processing step.
    """
    try:
        ymd, batch, repetition = extract_metadata_from_path(dir_path)
        input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
        logger.info(f"Starting background adder for {input_file}")
        config = get_configuration(input_file, logger)
        bg_id, dbg_id = get_background_identifiers(input_file, logger)
        # write result to HDF5 file: 
        with h5py.File(input_file, 'a') as h5f:
            bg_file_dataset = h5f.require_dataset('/entry1/processing_required_metadata/background_file', shape=(), dtype=h5py.special_dtype(vlen=str))
            rel_destination = ''
            if bg_id:
                destination = defaults.data_dir / bg_id[0:4] / bg_id[0:8] / f'MOUSE_{bg_id}_{config}_stacked.nxs'
                rel_destination = destination.relative_to(input_file.parent, walk_up=True)
            bg_file_dataset[...] = str(rel_destination)
            # bg_file_dataset[...] = str(Path('..', '..') / bg_id[0:4] / bg_id[0:8] / f'MOUSE_{bg_id}_{config}_stacked.nxs') if bg_id else ''
            dbg_file_dataset = h5f.require_dataset('/entry1/processing_required_metadata/dispersed_background_file', shape=(), dtype=h5py.special_dtype(vlen=str))
            rel_destination = ''
            if dbg_id:
                destination = defaults.data_dir / dbg_id[0:4] / dbg_id[0:8] / f'MOUSE_{dbg_id}_{config}_stacked.nxs'
                rel_destination = destination.relative_to(input_file.parent, walk_up=True)
            dbg_file_dataset[...] = str(rel_destination)
            # dbg_file_dataset[...] = str(Path('..', '..') / dbg_id[0:4] / dbg_id[0:8] / f'MOUSE_{dbg_id}_{config}_stacked.nxs') if dbg_id else ''
        logger.info(f"Completed background adder step for {input_file}")
    except Exception as e:
        # Print the standard output and standard error
        logger.info(f"Processstep processstep_add_background_files failed with stderr:")
        logger.info(e)
        logger.error(f"Error during background adder subprocess: {e}")
