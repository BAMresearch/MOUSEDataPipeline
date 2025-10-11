import h5py
import numpy as np
from pathlib import Path
import logging
from YMD_class import extract_metadata_from_path
from typing import List, Dict
from skimage import measure, morphology
from HDF5Translator.utils import Q_  # type: ignore


def get_float_from_h5(filename: Path, HDFPath: str, logger: logging.Logger) -> float:
    """
    Returns the value from the HDF5 file at HDFPath.
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            val = h5f[HDFPath][()]
        if isinstance(val, list) or isinstance(val, np.ndarray):
            val = np.mean(val)
        # else:
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


def get_pint_quantity_from_h5(filename: Path, HDFPath: str, logger: logging.Logger) -> Q_:
    """
    Returns the value with units as a Pint Quantity (Q_) from the HDF5 file at HDFPath.
    """
    try:
        val = get_float_from_h5(filename, HDFPath, logger)
        # get extra units information if available
        with h5py.File(filename, 'r') as h5f:
            units = h5f[HDFPath].attrs.get('units', 'dimensionless')
        if isinstance(units, bytes):
            units = units.decode('utf-8')
        pint_quantity = Q_(val, units)
    except Exception as e:
        logger.warning(f'could not read value with units {HDFPath} from {filename} with error {e}')
        pint_quantity = Q_(np.nan, 'dimensionless')
    return pint_quantity


def get_processed_files(dir_path: Path) -> List[Path]:
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    parent_path = dir_path.parent
    # print(parent_path)
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


def reduce_extra_image_dimensions(image: np.ndarray, method=np.mean) -> np.ndarray:
    assert method in [np.mean, np.sum, np.any, np.all], "method must be either np.mean or np.sum function handles"
    while image.ndim > 2:
        image = method(image, axis=0)
    return image


def prepare_eiger_image(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Prepares an Eiger image by removing masked or pegged pixels and reducing extra dimensions.
    """
    # Step 1: get rid of masked or pegged pixels on an Eiger detector
    labeled_foreground = (np.logical_and(image >= 0, image <= 2e7)).astype(int)
    maskedTwoDImage = image * labeled_foreground  # apply mask

    # Step 2: reduce extra dimensions if present
    reduced_image = reduce_extra_image_dimensions(maskedTwoDImage, method=np.mean)

    if reduced_image.ndim != 2:
        logger.error(f"Image could not be reduced to 2D, current shape: {reduced_image.shape}")
        raise ValueError("Image could not be reduced to 2D")

    return reduced_image


def label_main_feature(maskedTwoDImage: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Labels the main feature in the image using connected component analysis.
    """
    threshold_value = np.maximum(
        1, 1*maskedTwoDImage.mean()  # 0.0001 * maskedTwoDImage.max()
    )  # filters.threshold_otsu(maskedTwoDImage) # ignore zero pixels

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
        raise ValueError("No beam found in the image in call to label_main_feature.")
    if num > 1:
        logger.info("Multiple beams found in the image, selecting the largest one.")
        # find the largest component and keep only that one
        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # skip background label 0
        labels = (labels == largest_label).astype(int)
    # assert we only have one labeled region now
    assert np.unique(labels).size == 2, ValueError("More than one labeled region found in call to label_main_feature.")

    return labels
