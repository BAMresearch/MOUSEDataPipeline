from datetime import datetime
from pathlib import Path
import subprocess

import HDF5Translator
import h5py
import numpy as np
from YMD_class import YMD, extract_metadata_from_path
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging
from HDF5Translator.translator_elements import TranslationElement
from HDF5Translator.translator import process_translation_element
from pint import UnitRegistry

# Initialize the unit registry
ureg = UnitRegistry()

doc = """
WIP: This processing step updates the metadata in the translated and beam-analyzed files
with details from the logbook and project/sample information
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
        logger.info(f"metadata_updater cannot run in {dir_path}, file missing at: {step_2_file}")
        return False

    return True


def findentry(ymd:YMD, batch:int, logbook_reader: Logbook2MouseReader):
    # print(f'searching for {ymd.YMD} and {batch}, type {type(ymd.YMD)} and {type(batch)}')
    batch = int(batch)
    for entry in logbook_reader.entries:
        # print(f'checking {entry.ymd} and {entry.batchnum}, type {type(entry.ymd)} and {type(entry.batchnum)}')
        if entry.ymd == ymd.YMD and entry.batchnum == batch:
            return entry
    return None


def get_energy_from_h5(filename: Path, logger: logging.Logger) -> float:
    """
    Returns the energy value from the HDF5 file via wavelength and conversion
    """
    try: 
        with h5py.File(filename,'r') as h5f:
            wavelength = h5f['/entry1/sample/beam/incident_wavelength'][()]
            wavelength_units = h5f['/entry1/sample/beam/incident_wavelength'].attrs['units']
    except Exception as e:
        logger.warning(f'Could not read wavelength from {filename} with error {e}')
        return 0.0

    if isinstance(wavelength, (list, np.ndarray)):
        wavelength = np.mean(wavelength)

    if not isinstance(wavelength, np.floating):
        logger.warning(f'Incident wavelength not a float in {filename}')
        return 0.0

    if wavelength <= 0:
        logger.warning(f'Wavelength negative or zero in {filename}')
        return 0.0

    try:
        # Create a quantity with wavelength and its units
        wavelength_quantity = wavelength * ureg(wavelength_units)

        # Calculate energy using the formula E = hc/λ
        energy_quantity = (ureg.planck_constant * ureg.speed_of_light) / wavelength_quantity

        # Convert energy to keV
        energy_keV = energy_quantity.to('keV').magnitude
    except Exception as e:
        logger.warning(f'Error converting wavelength to energy in {filename} with error {e}')
        return 0.0

    return energy_keV


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Updates some metadata fields in the process with a new version of the metadata from the logbook and project/sample information.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    input_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}.nxs'
    # get the logbook entry for this measurement
    entry = findentry(ymd, batch, logbook_reader)
    energy = get_energy_from_h5(input_file, logger)
    # print(f'* * * * * * * * {energy=} * * * * * * * * ')
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
                destination='/entry1/processing_required_metadata/background_identifier',
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
                destination='/entry1/processing_required_metadata/dispersant_background_identifier',
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
                data_type="float",
                default_value=entry.matrixfraction,
                attributes={
                    "note": "The volume fraction that the matrix takes up in the total sample. For dilute samples, this approaches 1.0"
                },
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/sample/samplethickness',
                minimum_dimensionality=1,
                data_type="float",
                default_value=entry.samplethickness,
                attributes={
                    "note": "The thickness of the sample as specified in the measurement logbook in meters.",
                },
                source_units='m',
                destination_units='m',
            ),
            TranslationElement(
                destination='/entry1/sample/overall_mu',
                minimum_dimensionality=1,
                data_type="float",
                default_value=entry.sample.calculate_overall_properties(energy)['overall_mu'],
                attributes={
                    "note": "The volume fraction that the matrix takes up in the total sample. For dilute samples, this approaches 1.0",
                },
                source_units='1/m',
                destination_units='1/m'
            ),
            TranslationElement(
                # source is none since we're storing derived data
                destination='/entry1/sample/density',
                minimum_dimensionality=1,
                data_type="float",
                default_value=entry.sample.density,
                attributes={
                    "note": "Overall gravimetric density of the sample."
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
                default_value=entry.sampleposition['xsam'],
                attributes={
                    "depends_on": '.',
                    "offset": [0.0000000, 0.0000000, 0.0000000],
                    "offset_units": 'mm',
                    "transformation_type": 'translation',
                    "units": 'mm',
                    'vector': [0.0000000, 0.0000000, 1.0000000],
                    "note": "x offset of the sample, along the beam direction."
                },
                source_units='mm',
                destination_units='mm'
            ),   
        ]

        update_from_logbook = {
            '/entry1/processing_required_metadata/procpipeline': entry.procpipeline,
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
                    # attributes={
                    #     "note": "Auto-updated from the logbook by the post-translation processstep_metadata_update."
                    # },
                )
            )

        # let's try storing the sample components: 
        for component in entry.sample.components: # instance of SampleComponent
            TElements += [
                TranslationElement(
                    # source is none since we're storing derived data
                    destination=f'/entry1/sample/components/{component.component_id}/composition',
                    data_type="string",
                    default_value=component.composition,
                    attributes={
                        "note": "Atommic composition of the phase in the sample"
                    },
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/density',
                    data_type="float",
                    minimum_dimensionality=1,
                    default_value=component.density,
                    attributes={
                        "units": 'g/cm^3',
                        "note": "Density of the phase in the sample",
                    },
                    source_units='g/cm^3',
                    destination_units='g/cm^3'
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/mass_fraction',
                    data_type="float",
                    minimum_dimensionality=1,
                    default_value=component.mass_fraction,
                    attributes={
                        "note": "Mass fraction of the phase in the sample",
                    },
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/volume_fraction',
                    data_type="float",
                    minimum_dimensionality=1,
                    default_value=component.volume_fraction,
                    attributes={
                        "note": "Volume fraction of the phase in the sample",
                    },
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/name',
                    data_type="string",
                    default_value=component.name,
                    attributes={
                        "note": "Name of the phase in the sample",
                    },
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/connected_to',
                    data_type="string",
                    default_value=component.connected_to,
                    attributes={
                        "note": "phase connected to this other phase in the sample",
                    },
                ),
                TranslationElement(
                    destination=f'/entry1/sample/components/{component.component_id}/connection',
                    data_type="string",
                    default_value=component.connection,
                    attributes={
                        "note": "The way the phase is connected to this other phase in the sample",
                    },
                ),
            ]

        # writing the resulting metadata back to the main HDF5 file
        with h5py.File(input_file, "r+") as h5_out:
            for element in TElements:  # iterate over the two elements and write them back
                process_translation_element(None, h5_out, element)

        logger.info(f"Completed translator step for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e)
