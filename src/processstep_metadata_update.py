from datetime import datetime
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
WIP: This processing step updates the metadata in the translated and beam-analyzed files 
with details from the logbook and project/sample information
"""

# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step could run.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_2_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    if not step_2_file.is_file():
        logger.info(f"metadata_updater cannot run in {dir_path}, file missing at: {step_2_file}")
        return False

    return True

def findentry(ymd:YMD, batch:str, logbook_reader: Logbook2MouseReader):
    # print(f'searching for {ymd.YMD} and {batch}, type {type(ymd.YMD)} and {type(batch)}')
    for entry in logbook_reader.entries:
        # print(f'checking {entry.ymd} and {entry.batchnum}, type {type(entry.ymd)} and {type(entry.batchnum)}')
        if entry.ymd == ymd.YMD and entry.batchnum == batch:
            return entry
    return None

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Updates some metadata fields in the process with a new version of the metadata from the logbook and project/sample information.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    # get the logbook entry for this measurement
    entry = findentry(ymd, batch, logbook_reader)
    if entry is None:
        logger.info(f"metadata_updater cannot run for {dir_path}, no logbook entry found for {ymd=} and {batch=}")
        return

    # now we can update the metadata with the entry information
    # print(entry)
    try:
        logger.info(f"Starting metadata updater step for {input_file}")
        # This class lets you configure exactly what the output should look like in the HDF5 file.
        TElements = []  # we want to add two elements, so I make a list
        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/collection_identifier',
                minimum_dimensionality=1,
                data_type="string",
                default_value=f'{ymd.YMD}_{entry.batchnum}',
                attributes={
                    "note": "YMD and measurement group number"
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/processing_metadata/background_identifier',
                minimum_dimensionality=1,
                data_type="string",
                default_value=f'{entry.bgdate.year}{entry.bgdate.month:02d}{entry.bgdate.day:02d}_{entry.bgnumber}',
                attributes={
                    "note": "Instrument background YMD and measurement group number"
                },
            ),

        ]

        if (entry.dbgdate is not None) and (entry.dbgnumber is not None):
            dbgymd = f'{entry.dbgdate.year}{entry.dbgdate.month:02d}{entry.dbgdate.day:02d}'
            dbg_identifier = f'{dbgymd}_{entry.dbgnumber}'
        else: 
            dbg_identifier = 'None'

        TElements += [
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/processing_metadata/dispersant_background_identifier',
                minimum_dimensionality=1,
                data_type="string",
                default_value=dbg_identifier,
                attributes={
                    "note": "Optional dispersant background YMD and measurement group number. None if not set (i.e. for dilute analytes)."
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/sample/matrixfraction',
                minimum_dimensionality=1,
                data_type="string",
                default_value=f'{entry.matrixfraction}',
                attributes={
                    "note": "Updated by the post-translation processstep_metadata_update."
                },
            ),   
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/sample/density',
                minimum_dimensionality=1,
                data_type="float",
                default_value=f'{entry.sample.density}',
                attributes={
                    "note": "Updated by the post-translation processstep_metadata_update."
                },
                source_units='g/cm^3',
                destination_units='g/cm^3'
            ),   

            # this might get updated if we have a different sample offset in the beam direction.
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/sample/transformations/sample_x',
                minimum_dimensionality=1,
                data_type="float",
                default_value=f'{entry.sampleposition['xsam']}',
                attributes={
                    "depends_on" : '.',
                    "offset" : [0.0000000, 0.0000000, 0.0000000],
                    "offset_units": 'mm',
                    "transformation_type": 'translation',
                    "units": 'mm',
                    'vector': [0.0000000, 0.0000000, 1.0000000],
                    "note": "Updated by the post-translation processstep_metadata_update."
                },
                source_units='mm',
                destination_units='mm'
            ),   
        ]


        update_from_logbook = {
            '/entry1/experiment/procpipeline': entry.procpipeline,
            '/entry1/experiment/experiment_identifier': entry.proposal,
            '/entry1/experiment/user': entry.user,
            '/entry1/experiment/notes': entry.notes,
            '/entry1/experiment_identifier': entry.proposal,
            '/entry1/sample/sampleid': entry.sampleid,
            '/entry1/sample/name': entry.sample.sample_name,
            '/entry1/sample/composition': entry.sample.composition,
            '/entry1/proposal/proposalid': entry.proposal,
            '/entry1/proposal/proposal_title': entry.project.title,
            '/entry1/proposal/proposal_responsible_name': entry.project.name,
            '/entry1/proposal/proposal_responsible_email': entry.project.email,
            '/entry1/proposal/proposal_responsible_organisation': entry.project.organisation,
            # not sure if this will work for long descriptions, we might run our of string length... :
            '/entry1/proposal/proposal_description': entry.project.description,
            
            # '/entry1/sample/sample_name': entry.samplename,
        }
        for dest, value in update_from_logbook.items():
            TElements.append(
                TranslationElement(
                    # source is none since we're storing derived data
                    destination=dest,
                    minimum_dimensionality=1,
                    data_type="string",
                    default_value=value,
                    attributes={
                        "note": "Updated by the post-translation processstep_metadata_update."
                    },
                )
            )


        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)

        logger.info(f"Completed translator step for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e)
