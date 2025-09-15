import h5py
import numpy as np
from pathlib import Path
import logging
from YMD_class import extract_metadata_from_path
from typing import List, Dict


def get_float_from_h5(filename: Path, HDFPath: str, logger: logging.Logger) -> float:
    """
    Returns the value from the HDF5 file at HDFPath.
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            val = h5f[HDFPath][()]
        if isinstance(val, list) or isinstance(val, np.ndarray):
            val = np.mean(val)
        else:
            try:  # try to convert to float
                val = np.float32(val)
            except ValueError:
                logger.warning(f'could not convert {val} from {HDFPath} in {filename} to float')
                return 0.0
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


def get_processed_files(dir_path: Path) -> List[Path]:
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    parent_path = dir_path.parent
    print(parent_path)
    processed_files = list(parent_path.glob(f'{ymd.YMD}_{batch}_*/MOUSE_{ymd.YMD}_{batch}_*.nxs'))
    return processed_files


def sort_processed_files_by_instrument_configuration(processed_files: List[Path], logger: logging.Logger) -> Dict[str, List[Path]]:
    """
    Sorts the processed files by instrument configuration, as read from the processed files themselves.
    Outputs a dictionary with the instrument configuration as the key and a sorted list of processed files as the value.
    """
    config_to_files: Dict[str, List[Path]] = {}

    for f in processed_files:
        measurement_config = str(get_configuration(f, logger))
        
        # Add the file to the dictionary under the correct key
        if measurement_config not in config_to_files:
            config_to_files[measurement_config] = []
        
        config_to_files[measurement_config].append(f)

    # Sort each list of files by their modification time (optional)
    for config, files in config_to_files.items():
        # new sorting to make sure all filenames are sorted by repetition number
        config_to_files[config] = sorted(files, key=lambda f: int(f.stem.rsplit('_', 1)[-1]))
        # config_to_files[config] = sorted(files, key=lambda f: f.stat().st_mtime)

    return config_to_files


def get_configuration(filename: Path, logger: logging.Logger) -> int:
    """
    Read the configuration file and return the configuration
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            configuration = h5f['/entry1/instrument/configuration'][()]
    except Exception as e:
        logger.error(f"Error reading configuration from file: {e}")
        configuration = 0
    return int(configuration)