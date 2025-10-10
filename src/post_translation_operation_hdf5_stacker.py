#!/usr/bin/env python
# coding: utf-8

description = """
Post-translation HDF5 step for stacking datasets and metadata from 
multiple repetitions of a measurement. 

Usage:
  python post_translation_hdf5_stacker -k config=stacking_config.yaml --output measurement_stacked.h5 --auxiliary_files input_file1.h5 input_file2.h5

"""

import argparse
import h5py
import numpy as np
import yaml
import logging
from pathlib import Path

# from HDF5Translator.utils.data_utils import sanitize_attribute
from HDF5Translator.utils.validators import (
    validate_file, validate_file_delete_if_exists, validate_yaml_file
)
from HDF5Translator.utils.configure_logging import configure_logging

def canStack(filename:Path)->bool:
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
    # checklist for a few key critical items to ensure we've preprocessed correctly:
    checkList = [
        # "entry1/experiment/environment_temperature",
        # "entry1/experiment/stage_temperature",
        "entry1/instrument/detector00/data", # assure primary data is there

        "entry1/sample/beam/flux", # beam analysis has been done
        "entry1/sample/beam/incident_wavelength",
        # "entry1/sample/thickness", # thickness calculation has been entered from the beam analysis
        "entry1/sample/transmission", # beam analysis with both beams is there

        "entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",        
        # "entry1/processing/sample_beam_profile/beam_analysis/centerOfMass",        
    ]
    # check that the filenames referenced in these paths exist:
    checkFileExistence = [
        # background file cannot be checked at this stage as it might not exist yet. :(
        # "entry1/processing_required_metadata/background_file", 
        "entry1/processing_required_metadata/mask_file", 
    ]

    with h5py.File(filename, 'r') as h5f:
        try:
            for path in checkList:
                if not path in h5f:
                    logging.warning(f'path not found: {path} in file {filename}')
                    return False

            for path in checkFileExistence:
                if not path in h5f:
                    logging.warning(f'path not found: {path} in file {filename}')
                    return False
                full_path = Path(filename.parent, h5f[path][()].decode('utf-8')).resolve() # relative paths
                if not full_path.is_file():
                    logging.warning(f'file {h5f[path][()].decode('utf-8')} not found at: {path} in file {filename}')
                    return False

        except Exception as e:
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
            outputFile:Path = None, 
            filenames:list = [], 
            stackItems:list = [], 
            calculate_average:list = [],
            adjust_relative_path_oneup:list = []
            ):
        assert isinstance(outputFile, Path), 'output filename must be a path instance'
        assert len(filenames) > 0, 'at least one file is required for stacking.'
        # assert that the filenames to stack all exist:
        okFilenames = filenames.copy()
        for fname in filenames:
            # if the file does not pass the canStack test, remove it from the list:
            if not canStack(fname):
                okFilenames.remove(fname)
                logging.warning(f'file {fname} does not pass the canStack test, removing from list of files to stack.')
                # save the file in an error list text file:
                with open(outputFile.with_suffix('.stacking_error_list'), 'a') as f:
                    f.write(f'{fname}\n')
            assert fname.exists(), f'filename {fname} does not exist in the list of files to stack.'
        assert len(okFilenames) > 0, 'after checking, not enough valid files for stacking.'
        # store the filenames that passed the canStack test:
        filenames = okFilenames
        # store in the class
        self.outputFile = outputFile
        self.stackItems = stackItems
        self.filenames = filenames

        # use the first file as a template, increasing the size of the datasets to stack

        self.createStructureFromFile(filenames[0], addShape = (len(filenames),)) # addShape = (len(filenames), 1)

        # add the datasets to the file.. this could perhaps be done in parallel
        for idx, filename in enumerate(filenames): 
            # print(f'adding file {idx+1} of {len(filenames)}: {filename}')
            self.addDataToStack(filename, addAtStackLocation = idx)

        # now we calculate the mean, std and standard error on the mean of selected datasets:
        for path in calculate_average:
            self.calculateAverage(path)

        for path in adjust_relative_path_oneup:
            self.adjustRelativePath(path)
        
    def adjustRelativePath(self, path):
        """
        adjusts the relative paths in the location to be one level up, 
        e.g. "../../Mask/file.nxs" becomes "../Mask/file.nxs"        
        """
        with h5py.File(self.outputFile, 'a') as h5out:
            if not path in h5out:
                logging.warning(f'path {path} not found in output file, skipping')    
                return

            oldPath = h5out[path][()]
            if oldPath is None:
                logging.debug(f'path {path} is empty, skipping')
                return
            if isinstance(oldPath, bytes):
                oldPath = oldPath.decode('utf-8')

            if oldPath == '':
                logging.debug(f'path {path} is empty, skipping')
                return
            oldPath = Path(h5out[path][()].decode('utf-8'))
            try:
                newPath = oldPath.relative_to('..')
            except ValueError:
                logging.warning(f'path {path} already at root level or cannot be made relative to parent, skipping')
                return
            h5out[path][...] = str(newPath)

    def calculateAverage(self, path):
        with h5py.File(self.outputFile, 'a') as h5out:
            if path in h5out:
                logging.debug(f'calculating average for path: {path}')
                data = h5out[path][()]
                # assure data is an array with dtype float
                data = np.array(data, dtype=float)
                attributes = h5out[path].attrs
                newattrs = {k: attributes[k] for k in attributes.keys()}
                # make sure there's a note in newattrs: 
                if "note" not in newattrs:
                    newattrs["note"] = ""
                newattrs['note'] = newattrs["note"] + " averaged for repetitions using post_translation_hdf5_stacker.py"
                ds = h5out.create_dataset(f'{path}_averaged/mean', data=data.mean())
                ds.attrs.update(newattrs)
                ds = h5out.create_dataset(f'{path}_averaged/std', data=data.std(ddof=1))
                ds.attrs.update(newattrs)
                ds = h5out.create_dataset(f'{path}_averaged/sem', data=data.std(ddof=1) / np.sqrt(np.size(data)))
                ds.attrs.update(newattrs)
                ds = h5out.create_dataset(f'{path}_averaged/max', data=data.max())
                ds.attrs.update(newattrs)
                ds = h5out.create_dataset(f'{path}_averaged/min', data=data.min())
                ds.attrs.update(newattrs)
                ds = h5out.create_dataset(f'{path}_averaged/count', data=np.size(data))
                newattrs.update({'units': "dimensionless"})
                ds.attrs.update(newattrs)  # count has no units
            else:
                logging.warning(f'path {path} not found in output file, skipping average calculation')

    def createStructureFromFile(self, ifname, addShape):
        """addShape is a tuple with the dimensions to add to the normal datasets. i.e. (280, 1) will add those dimensions to the array shape"""
        # input = nx.nxload(ifname)
        with h5py.File(ifname, 'r') as h5in, h5py.File(self.outputFile, 'w') as h5out:
            # using h5py.visititems to walk the file

            def printLinkItem(name, obj):
                logging.debug(f'Link item found: {name= }, {obj= }')

            def addItem(name, obj):
                if 'entry1/instrument/detector/detectorSpecific' in name:
                    logging.debug(f'found the path: {name} in file {ifname}')                
                if isinstance(obj, h5py.Group):
                    logging.debug(f'adding group: {name}')
                    h5out.create_group(name)
                    # add attributes
                    h5out[name].attrs.update(obj.attrs)
                elif isinstance(obj, h5py.Dataset) and not (name in self.stackItems):
                    logging.debug(f'plainly adding dataset: {name}')
                    h5in.copy(name, h5out, expand_external=True, name=name)
                    h5out[name].attrs.update(obj.attrs)
                    # h5out.create_dataset(name, data=obj[()])
                elif isinstance(obj, h5py.Dataset) and name in self.stackItems:
                    logging.debug(f'preparing by initializing the stacked dataset: {name} to shape {(*addShape, *obj.shape)}')
                    totalShape = (*addShape, *obj.shape)
                    chunkShape = list(totalShape)
                    chunkShape[0] = 1
                    h5out.create_dataset(
                        name,
                        shape = totalShape,
                        chunks = tuple(chunkShape),
                        maxshape = totalShape,
                        dtype = obj.dtype,
                        compression="gzip", # if we switch on gzip compression, this operation becomes very slow
                        # data = obj[()]
                    )
                    h5out[name].attrs.update(obj.attrs)
                else:
                    logging.info(f'** uncaught object: {name}')
            
            h5in.visititems(addItem)
            h5in.visititems_links(printLinkItem)

    def addDataToStack(self, ifname, addAtStackLocation):
        with h5py.File(ifname, 'r') as h5in, h5py.File(self.outputFile, 'a') as h5out:
            for path in self.stackItems:
                if path in h5in and path in h5out:
                    logging.debug(f'adding data to stack: {path} at stackLocation: {addAtStackLocation}')
                    # print(f'adding data to stack: {path} at stackLocation: {addAtStackLocation}')
                    h5out[path][addAtStackLocation] = h5in[path][()]            
                elif not path in h5in:
                    logging.warning(f'** could not find path {path} in input file,. skipping...')
                elif not path in h5out:
                    logging.warning(f'** could not find path {path} in output file, skipping...')
                else:
                    logging.warning(f'** uncaught error with path {path}, skipping...')


# If you are adjusting the template for your needs, you probably only need to touch the main function:
def main(
    output: Path,
    auxiliary_files: list[Path],
    config: Path,
):
    """

    """
    # Process input parameters:
    # Make sure we have at least two files to stack, something argparse cannot do
    assert len(auxiliary_files) >= 1, "At least one file is required for stacking."

    # read the stacking section of the configuration file, which contains two sections: which datasets to stack and which to calculate the average and standard deviation over:
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        stack_datasets = config.get("stack_datasets", None)
        calculate_average = config.get("calculate_average", None)
        adjust_relative_path_oneup = config.get("adjust_relative_path_oneup", None)
    # at least the stack_datasets dictionary must exist: 
    assert stack_datasets is not None, "The configuration file must contain a 'stack_datasets' section."
    # Stack the datasets
    newNewConcat(output, auxiliary_files, stack_datasets, calculate_average, adjust_relative_path_oneup)

    logging.info("Post-translation processing complete.")


def setup_argparser():
    """
    Sets up command line argument parser using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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

    logging.info(f"Stacking into new file: {args.output}")
    logging.info(f"with configuration file: {args.config}")
    if args.auxiliary_files:
        for auxiliary_file in args.auxiliary_files:
            logging.info(f"stacking source file: {auxiliary_file}")

    main(args.output, args.auxiliary_files, args.config)
