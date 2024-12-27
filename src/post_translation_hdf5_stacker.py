#!/usr/bin/env python
# coding: utf-8

description = """
Post-translation HDF5 step for stacking datasets and metadata from 
multiple repetitions of a measurement. 

Usage:
  python post_translation_hdf5_stacker -k config=stacking_config.yaml --output measurement_stacked.h5 --auxiliary_files input_file1.h5 input_file2.h5

"""

import argparse
import os
import sys
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
        assert len(filenames) > 2, 'at least two files are required for stacking.'
        # assert that the filenames to stack all exist:
        for fname in filenames:
            assert fname.exists(), f'filename {fname} does not exist in the list of files to stack.'

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


class nexConcat(object):
    # """
    # Concatenates multiple NeXus files *with identical structure* to frames of a single file. 
    # Useful for combining multiple exposures, structurized using NeXpand, into a series of frames.
    
    # All data in a single array in the input nexus files are concatenated along axis NeXusAxis. 
    # Non-array values are read from the first file and stored in the new file. 
    # """

    inflist = None
    ofname = None
    remove = True
    bDict = {}
    allItems = []  # is filled in by self.listStructure
    # forceDType = {"entry1/frames/data/data_000001": np.float64}
    # stackItems = [
    # ...  - now in YAML file..
    # ]

    def clear(self):
        self.inflist = []
        self.stackItems=[],
        self.ofname = None
        self.remove = True
        self.bDict = {}
        self.allItems = []

    def __init__(
        self,
        inflist:list=[],
        ofname:Path=None,
        remove=False,
        stackItems:list=[],
        NeXpandScriptVersion=None,
    ):
        self.clear()
        self.inflist = inflist
        self.ofname = ofname
        self.remove = remove
        self.NeXpandScriptVersion = NeXpandScriptVersion

        # delete output file if exists, and requested by input flag
        if os.path.exists(self.ofname) and self.remove:
            logging.info("file {} exists, removing...".format(self.ofname))
            os.remove(self.ofname)

        for infile in self.inflist:
            self.expandBDict(infile)

        self.listStructure(self.inflist[0])
        # remove stackItems from that list
        for x in self.stackItems:
            # print("removing item {} from self.allitems: {} \n \n".format(x, self.allItems))
            try:
                self.allItems.remove(x)
            except ValueError:
                logging.info("Item {} not found in allItems, skipping...".format(x))

        self.hdfDeepCopy(self.inflist[0], self.ofname)
        with h5py.File(self.ofname, "a") as h5f:
            # del h5f["Saxslab"] not the issue here.
            h5f["/"].attrs["default"] = "entry1"

    def _stackIt(self, name, obj):
        # print(name)
        if name in self.stackItems:
            if name in self.bDict:  # already exists:
                try:
                    self.bDict[name].append(
                        obj[()]
                    )  # np.stack((self.bDict[name], obj[()]))

                except ValueError:
                    logging.warning(
                        "\n\n file: {} \n NeXus path: {} \n value: {}".format(
                            self.ofname, name, obj[()]
                        )
                    )
                    logging.warning(
                        "\n\n bDict[name]: {} \n bDict[name] shape: {} \n value shape: {}".format(
                            self.bDict[name], self.bDict[name].shape, obj[()].shape
                        )
                    )
                    raise
                except:
                    raise
            else:  # create entry
                self.bDict[name] = [obj[()]]  # list of entries

    def expandBDict(self, filename):

        with h5py.File(filename, "r") as h5f:

            h5f.visititems(self._stackIt)

    def hdfDeepCopy(self, ifname, ofname):
        """Copies the internals of an open HDF5 file object (infile) to a second file, 
        replacing the content with stacked data where necessary..."""
        with h5py.File(ofname, "a") as h5o, h5py.File(
            ifname, "r"
        ) as h5f:  # create file, truncate if exists
            # first copy stacked items, adding attributes from infname
            for nxPath in self.stackItems:
                gObj = h5f.get(nxPath, default=None)
                if gObj is None:
                    # print("Path {} not present in input file {}".format(nxPath, ifname))
                    return

                # print("Creating dataset at path {}".format(nxPath))
                # if nxPath in self.forceDType.keys():
                #     logging.debug("forcing dType {}".format(self.forceDType[nxPath]))
                #     oObj = h5o.create_dataset(
                #         nxPath,
                #         data=np.stack(self.bDict[nxPath]),
                #         dtype=self.forceDType[nxPath],
                #         compression="gzip",
                #     )
                # else:
                oObj = h5o.create_dataset(
                    nxPath, data=np.stack(self.bDict[nxPath]), compression="gzip"
                )
                oObj.attrs.update(gObj.attrs)

            # now copy the rest, skipping what is already there.
            for nxPath in self.allItems:
                # do not copy groups...
                oObj = h5o.get(nxPath, default="NonExistentGroup")
                if isinstance(oObj, h5py.Group):
                    # skip copying the group, but ensure all attributes are there..
                    gObj = h5f.get(nxPath, default=None)
                    if gObj is not None:
                        oObj.attrs.update(gObj.attrs)
                    continue  # skippit

                groupLoc = nxPath.rsplit("/", maxsplit=1)[0]
                if len(groupLoc) > 0:
                    gl = h5o.require_group(groupLoc)
                    # copy group attributes
                    oObj = h5f.get(groupLoc)
                    gl.attrs.update(oObj.attrs)

                    try:
                        h5f.copy(nxPath, gl)
                    except (RuntimeError, ValueError):
                        pass
                        # print("Skipping path {}, already exists...".format(nxPath))
                    except:
                        raise

    def listStructure(self, ifname):
        """ reads all the paths to the items into a list """

        def addName(name):
            self.allItems += [name]

        with h5py.File(ifname, "r") as h5f:
            h5f.visit(addName)
        # print(self.allItems)


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
    assert len(auxiliary_files) >= 2, "At least two files are required for stacking."

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
