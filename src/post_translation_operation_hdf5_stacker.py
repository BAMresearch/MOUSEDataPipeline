#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import yaml
from HDF5Translator.utils.configure_logging import configure_logging

# from HDF5Translator.utils.data_utils import sanitize_attribute
from HDF5Translator.utils.validators import validate_file, validate_file_delete_if_exists, validate_yaml_file

description = """
Post-translation HDF5 step for stacking datasets and metadata from
multiple repetitions of a measurement.

Usage:
  python post_translation_hdf5_stacker -k config=stacking_config.yaml --output measurement_stacked.h5 --auxiliary_files input_file1.h5 input_file2.h5

"""

LOGGER = logging.getLogger(__name__)
PRIMARY_DATA_PATH = "entry1/instrument/detector00/data"
DEFAULT_STACK_COMPRESSION = "lzf"
SUPPORTED_STACK_COMPRESSIONS = {"none": None, "lzf": "lzf", "gzip": "gzip"}


def normalize_stack_compression(compression: str | None) -> str | None:
    if compression is None:
        return SUPPORTED_STACK_COMPRESSIONS[DEFAULT_STACK_COMPRESSION]
    normalized = str(compression).strip().lower()
    if normalized not in SUPPORTED_STACK_COMPRESSIONS:
        raise ValueError(
            f"unsupported stack compression {compression!r}; expected one of {', '.join(SUPPORTED_STACK_COMPRESSIONS)}"
        )
    return SUPPORTED_STACK_COMPRESSIONS[normalized]


def canStack(filename: Path, logger: logging.Logger | None = None) -> bool:
    """
    Check if a file can be stacked.
    Parameters
    ----------
    filename : Path
        The path of the file to check.
    Returns
    -------
    bool
        True if the file can be stacked.
    """
    logger = logger or LOGGER
    # checklist for a few key critical items to ensure we've preprocessed correctly:
    checkList = [
        # "entry1/experiment/environment_temperature",
        # "entry1/experiment/stage_temperature",
        PRIMARY_DATA_PATH,  # assure primary data is there
        "entry1/sample/beam/flux",  # beam analysis has been done
        "entry1/sample/beam/incident_wavelength",
        # "entry1/sample/thickness", # thickness calculation has been entered from the beam analysis
        "entry1/sample/transmission",  # beam analysis with both beams is there
        "entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",
    ]
    # check that the filenames referenced in these paths exist:
    checkFileExistence = [
        # background file cannot be checked at this stage as it might not exist yet. :(
        # "entry1/processing_required_metadata/background_file",
        "entry1/processing_required_metadata/mask_file",
    ]

    with h5py.File(filename, "r") as h5f:
        try:
            for path in checkList:
                if path not in h5f:
                    logger.warning(f"path not found: {path} in file {filename}")
                    return False

            for path in checkFileExistence:
                if path not in h5f:
                    logger.warning(f"path not found: {path} in file {filename}")
                    return False
                full_path = Path(filename.parent, h5f[path][()].decode("utf-8")).resolve()  # relative paths
                if not full_path.is_file():
                    logger.warning(f"file {h5f[path][()].decode('utf-8')} not found at: {path} in file {filename}")
                    return False

        except Exception:
            return False

    return True


class newNewConcat(object):
    """
    Similar in structure to newConcat, but using h5py instead of nexusformat.nx
    """

    outputFile = None
    filenames = None
    core = None
    stackItems = None

    def __init__(
        self,
        outputFile: Path | None = None,
        filenames: list | None = None,
        stackItems: list | None = None,
        calculate_average: list | None = None,
        adjust_relative_path_oneup: list | None = None,
        match_detector_data_rank: bool = False,
        compression: str | None = None,
        logger: logging.Logger | None = None,
    ):
        if not isinstance(outputFile, Path):
            raise TypeError("output filename must be a path instance")

        self.logger = logger or LOGGER
        filenames = list(filenames or [])
        stackItems = list(stackItems or [])
        calculate_average = list(calculate_average or [])
        adjust_relative_path_oneup = list(adjust_relative_path_oneup or [])

        if len(filenames) == 0:
            raise ValueError("at least one file is required for stacking")

        # Check that the filenames to stack all exist:
        okFilenames = filenames.copy()
        for fname in filenames:
            if not fname.exists():
                raise FileNotFoundError(f"filename {fname} does not exist in the list of files to stack")
            # if the file does not pass the canStack test, remove it from the list:
            if not canStack(fname, logger=self.logger):
                okFilenames.remove(fname)
                self.logger.warning(
                    f"file {fname} does not pass the canStack test, removing from list of files to stack."
                )
                # save the file in an error list text file:
                with open(outputFile.with_suffix(".stacking_error_list"), "a", encoding="utf-8") as f:
                    f.write(f"{fname}\n")
        if len(okFilenames) == 0:
            raise ValueError("after checking, not enough valid files for stacking")

        # store the filenames that passed the canStack test:
        filenames = okFilenames
        # store in the class
        self.outputFile = outputFile
        self.stackItems = stackItems
        self.filenames = filenames
        self.match_detector_data_rank = match_detector_data_rank
        self.target_stacked_rank: int | None = None
        self.compression = normalize_stack_compression(compression)

        # use the first file as a template, increasing the size of the datasets to stack

        self.createStructureFromFile(filenames[0], addShape=(len(filenames),))  # addShape = (len(filenames), 1)

        # add the datasets to the file.. this could perhaps be done in parallel
        with h5py.File(self.outputFile, "a") as h5out:
            for idx, filename in enumerate(filenames):
                with h5py.File(filename, "r") as h5in:
                    self.addDataToStack(h5in, h5out, addAtStackLocation=idx)

        # now we calculate the mean, std and standard error on the mean of selected datasets:
        with h5py.File(self.outputFile, "a") as h5out:
            for path in calculate_average:
                self.calculateAverage(h5out, path)

            for path in adjust_relative_path_oneup:
                self.adjustRelativePath(h5out, path)

    def adjustRelativePath(self, h5out: h5py.File, path):
        """
        adjusts the relative paths in the location to be one level up,
        e.g. "../../Mask/file.nxs" becomes "../Mask/file.nxs"
        """
        if path not in h5out:
            self.logger.warning(f"path {path} not found in output file, skipping")
            return

        oldPath = h5out[path][()]
        if oldPath is None:
            self.logger.debug(f"path {path} is empty, skipping")
            return
        if isinstance(oldPath, bytes):
            oldPath = oldPath.decode("utf-8")

        if oldPath == "":
            self.logger.debug(f"path {path} is empty, skipping")
            return
        oldPath = Path(oldPath)
        try:
            newPath = oldPath.relative_to("..")
        except ValueError:
            self.logger.warning(f"path {path} already at root level or cannot be made relative to parent, skipping")
            return
        h5out[path][...] = str(newPath)

    def calculateAverage(self, h5out: h5py.File, path):
        if path in h5out:
            self.logger.debug(f"calculating average for path: {path}")
            data = h5out[path][()]
            # assure data is an array with dtype float
            data = np.array(data, dtype=float)
            attributes = h5out[path].attrs
            newattrs = {k: attributes[k] for k in attributes.keys()}
            # make sure there's a note in newattrs:
            if "note" not in newattrs:
                newattrs["note"] = ""
            newattrs["note"] = newattrs["note"] + " averaged for repetitions using post_translation_hdf5_stacker.py"
            ds = h5out.create_dataset(f"{path}_averaged/mean", data=data.mean())
            ds.attrs.update(newattrs)
            ds = h5out.create_dataset(f"{path}_averaged/std", data=data.std(ddof=1))
            ds.attrs.update(newattrs)
            ds = h5out.create_dataset(f"{path}_averaged/sem", data=data.std(ddof=1) / np.sqrt(np.size(data)))
            ds.attrs.update(newattrs)
            ds = h5out.create_dataset(f"{path}_averaged/max", data=data.max())
            ds.attrs.update(newattrs)
            ds = h5out.create_dataset(f"{path}_averaged/min", data=data.min())
            ds.attrs.update(newattrs)
            ds = h5out.create_dataset(f"{path}_averaged/count", data=np.size(data))
            newattrs.update({"units": "dimensionless"})
            ds.attrs.update(newattrs)  # count has no units
        else:
            self.logger.warning(f"path {path} not found in output file, skipping average calculation")

    def createStructureFromFile(self, ifname, addShape):
        """addShape is a tuple with the dimensions to add to the normal datasets. i.e. (280, 1) will add those dimensions to the array shape"""
        # input = nx.nxload(ifname)
        with h5py.File(ifname, "r") as h5in, h5py.File(self.outputFile, "w") as h5out:
            if self.match_detector_data_rank:
                if PRIMARY_DATA_PATH not in h5in:
                    raise ValueError(f"primary data path {PRIMARY_DATA_PATH} not found in template file {ifname}")
                self.target_stacked_rank = len(addShape) + len(h5in[PRIMARY_DATA_PATH].shape)

            # using h5py.visititems to walk the file

            def addItem(name, obj):
                if "entry1/instrument/detector/detectorSpecific" in name:
                    self.logger.debug(f"found the path: {name} in file {ifname}")
                if isinstance(obj, h5py.Group):
                    self.logger.debug(f"adding group: {name}")
                    h5out.create_group(name)
                    # add attributes
                    h5out[name].attrs.update(obj.attrs)
                elif isinstance(obj, h5py.Dataset) and name not in self.stackItems:
                    self.logger.debug(f"plainly adding dataset: {name}")
                    h5in.copy(name, h5out, expand_external=True, name=name)
                    h5out[name].attrs.update(obj.attrs)
                    # h5out.create_dataset(name, data=obj[()])
                elif isinstance(obj, h5py.Dataset) and name in self.stackItems:
                    self.logger.debug(
                        f"preparing by initializing the stacked dataset: {name} to shape "
                        f"{self._stacked_output_shape(obj.shape, addShape)}"
                    )
                    totalShape = self._stacked_output_shape(obj.shape, addShape)
                    dataset_kwargs = {
                        "shape": totalShape,
                        "maxshape": totalShape,
                        "dtype": obj.dtype,
                    }
                    if self.compression is not None:
                        chunkShape = list(totalShape)
                        chunkShape[0] = 1
                        dataset_kwargs["chunks"] = tuple(chunkShape)
                        dataset_kwargs["compression"] = self.compression
                    h5out.create_dataset(name, **dataset_kwargs)
                    h5out[name].attrs.update(obj.attrs)
                else:
                    self.logger.info(f"** uncaught object: {name}")

            h5in.visititems(addItem)
            if self.logger.isEnabledFor(logging.DEBUG):
                h5in.visititems_links(lambda name, obj: self.logger.debug(f"Link item found: {name= }, {obj= }"))

    def _stacked_output_shape(self, dataset_shape: tuple[int, ...], addShape: tuple[int, ...]) -> tuple[int, ...]:
        totalShape = (*addShape, *dataset_shape)
        if self.target_stacked_rank is None or len(totalShape) >= self.target_stacked_rank:
            return totalShape
        return (*totalShape, *((1,) * (self.target_stacked_rank - len(totalShape))))

    def _reshape_data_for_output(self, data, target_shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(data)
        reshape_shape = (*array.shape, *((1,) * (len(target_shape) - array.ndim)))
        try:
            reshaped = array.reshape(reshape_shape)
        except ValueError as exc:
            raise ValueError(
                f"could not reshape stacked dataset value from {array.shape} to target shape {target_shape}"
            ) from exc
        if reshaped.shape != target_shape:
            raise ValueError(f"stacked dataset value has incompatible shape {reshaped.shape}; expected {target_shape}")
        return reshaped

    def addDataToStack(self, h5in: h5py.File, h5out: h5py.File, addAtStackLocation):
        for path in self.stackItems:
            if path in h5in and path in h5out:
                self.logger.debug(f"adding data to stack: {path} at stackLocation: {addAtStackLocation}")
                data = h5in[path][()]
                target_shape = h5out[path][addAtStackLocation].shape
                if np.shape(data) != target_shape:
                    data = self._reshape_data_for_output(data, target_shape)
                h5out[path][addAtStackLocation] = data
            elif path not in h5in:
                self.logger.warning(f"** could not find path {path} in input file,. skipping...")
            elif path not in h5out:
                self.logger.warning(f"** could not find path {path} in output file, skipping...")
            else:
                self.logger.warning(f"** uncaught error with path {path}, skipping...")


# If you are adjusting the template for your needs, you probably only need to touch the main function:
def main(
    output: Path,
    auxiliary_files: list[Path],
    config: Path,
    match_detector_data_rank: bool = False,
    compression: str | None = None,
    logger: logging.Logger | None = None,
):
    """ """
    logger = logger or LOGGER
    # Process input parameters:
    # Make sure we have at least two files to stack, something argparse cannot do
    if len(auxiliary_files) < 1:
        raise ValueError("At least one file is required for stacking.")

    # read the stacking section of the configuration file, which contains two sections: which datasets to stack and which to calculate the average and standard deviation over:
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        stack_datasets = config.get("stack_datasets", None)
        calculate_average = config.get("calculate_average", None)
        adjust_relative_path_oneup = config.get("adjust_relative_path_oneup", None)
        configured_compression = config.get("compression", None)
    # at least the stack_datasets dictionary must exist:
    if stack_datasets is None:
        raise ValueError("The configuration file must contain a 'stack_datasets' section.")

    # Stack the datasets
    newNewConcat(
        output,
        auxiliary_files,
        stack_datasets,
        calculate_average,
        adjust_relative_path_oneup,
        match_detector_data_rank=match_detector_data_rank,
        compression=compression if compression is not None else configured_compression,
        logger=logger,
    )

    logger.info("Post-translation processing complete.")


def setup_argparser():
    """
    Sets up command line argument parser using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-o",
        "--output",
        type=validate_file_delete_if_exists,
        required=True,
        help="Output stacked measurement HDF5 file. Will be deleted if already existing.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=validate_yaml_file,
        required=True,
        help="stacker configuration YAML file.",
    )
    parser.add_argument(
        "-a",
        "--auxiliary_files",
        type=validate_file,
        required=True,
        nargs="+",
        help="HDF5 files to stack, at least two. (read-only)",
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
        "--match-detector-data-rank",
        action="store_true",
        help=(
            "Pad stacked datasets with trailing singleton dimensions until they have the same rank as "
            f"{PRIMARY_DATA_PATH}."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=tuple(SUPPORTED_STACK_COMPRESSIONS),
        default=None,
        help=(
            "Compression for stacked datasets. "
            f"Defaults to {DEFAULT_STACK_COMPRESSION!r}. Use 'none' for maximum write speed."
        ),
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
        log_file_prepend="PostTranslation_stacker_",
    )

    LOGGER.info(f"Stacking into new file: {args.output}")
    LOGGER.info(f"with configuration file: {args.config}")
    if args.auxiliary_files:
        for auxiliary_file in args.auxiliary_files:
            LOGGER.info(f"stacking source file: {auxiliary_file}")

    main(
        args.output,
        args.auxiliary_files,
        args.config,
        match_detector_data_rank=args.match_detector_data_rank,
        compression=args.compression,
        logger=LOGGER,
    )
