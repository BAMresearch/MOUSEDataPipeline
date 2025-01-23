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
        "entry1/experiment/environment_temperature",
        "entry1/experiment/stage_temperature",
        "entry1/instrument/detector00/data", # assure primary data is there

        "entry1/sample/beam/flux", # beam analysis has been done
        "entry1/sample/beam/incident_wavelength",
        "entry1/sample/thickness", # thickness calculation has been entered from the beam analysis
        "entry1/sample/transmission", # beam analysis with both beams is there

        "entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",        
        "entry1/processing/sample_beam_profile/beam_analysis/centerOfMass",        
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
                    return False

            for path in checkFileExistence:
                if not path in h5f:
                    return False
                if not Path(h5f[path][()].decode('utf-8')).is_file():
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

    def __init__(self, outputFile:Path = None, filenames:list = [], stackItems:list = []):
        assert isinstance(outputFile, Path), 'output filename must be a path instance'
        assert len(filenames) > 0, 'at least one file is required for stacking.'
        # assert that the filenames to stack all exist:
        for fname in filenames:
            # if the file does not pass the canStack test, remove it from the list:
            if not canStack(fname):
                filenames.remove(fname)
                logging.warning(f'file {fname} does not pass the canStack test, removing from list of files to stack.')
                # save the file in an error list text file:
                with open(outputFile.with_suffix('.stacking_error_list'), 'a') as f:
                    f.write(f'{fname}\n')
            assert fname.exists(), f'filename {fname} does not exist in the list of files to stack.'
        assert len(filenames) > 0, 'after checking, not enough valid files for stacking.'
 
        # store in the class
        self.outputFile = outputFile
        self.stackItems = stackItems
        self.filenames = filenames

        # use the first file as a template, increasing the size of the datasets to stack

        self.createStructureFromFile(filenames[0], addShape = (len(filenames),)) # addShape = (len(filenames), 1)

        # add the datasets to the file.. this could perhaps be done in parallel
        for idx, filename in enumerate(filenames): 
            self.addDataToStack(filename, addAtStackLocation = idx)


    def createStructureFromFile(self, ifname, addShape):
        """addShape is a tuple with the dimensions to add to the normal datasets. i.e. (280, 1) will add those dimensions to the array shape"""
        # input = nx.nxload(ifname)
        with h5py.File(ifname, 'r') as h5in, h5py.File(self.outputFile, 'w') as h5out:
            # using h5py.visititems to walk the file

            def printLinkItem(name, obj):
                print(f'Link item found: {name= }, {obj= }')

            def addItem(name, obj):
                if name == 'entry1/sample/beam/incident_wavelength':
                    print(f'found the path: {name}')                
                if isinstance(obj, h5py.Group):
                    print(f'adding group: {name}')
                    h5out.create_group(name)
                    # add attributes
                    h5out[name].attrs.update(obj.attrs)
                elif isinstance(obj, h5py.Dataset) and not (name in self.stackItems):
                    print(f'plainly adding dataset: {name}')
                    h5in.copy(name, h5out, expand_external=True, name=name)
                    h5out[name].attrs.update(obj.attrs)
                    # h5out.create_dataset(name, data=obj[()])
                elif isinstance(obj, h5py.Dataset) and name in self.stackItems:
                    print(f'preparing by initializing the stacked dataset: {name} to shape {(*addShape, *obj.shape)}')
                    h5out.create_dataset(
                        name,
                        shape = (*addShape, *obj.shape),
                        maxshape = (*addShape, *obj.shape),
                        dtype = obj.dtype,
                        compression="gzip",
                        # data = obj[()]
                    )
                    h5out[name].attrs.update(obj.attrs)
                else:
                    print(f'** uncaught object: {name}')
            
            h5in.visititems(addItem)
            h5in.visititems_links(printLinkItem)

    def addDataToStack(self, ifname, addAtStackLocation):
        with h5py.File(ifname, 'r') as h5in, h5py.File(self.outputFile, 'a') as h5out:
            for path in self.stackItems:
                if path in h5in and path in h5out:
                    print(f'adding data to stack: {path} at stackLocation: {addAtStackLocation}')
                    h5out[path][addAtStackLocation] = h5in[path][()]            
                elif not path in h5in:
                    print(f'** could not find path {path} in input file,. skipping...')
                elif not path in h5out:
                    print(f'** could not find path {path} in output file, skipping...')
                else:
                    print(f'** uncaught error with path {path}, skipping...')


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
        # calculate_average = config.get("calculate_average", None)
        
    # at least the stack_datasets dictionary must exist: 
    assert stack_datasets is not None, "The configuration file must contain a 'stack_datasets' section."
    # Stack the datasets
    newNewConcat(output, auxiliary_files, stack_datasets)

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
