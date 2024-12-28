# MOUSEDataPipeline
Tools for (automatic) processing of the new MOUSE datafiles. 

# prerequisites and assumptions

## Nomenclature
### Measurement Date
This is a rough datestamp on when the measurements on a particular set of samples were started. 
As mentioned, the rough idea is that each set of samples that belong together go in their own 
measurement date. The format of this is YYYYMMDD. 

### Batch
A batch is a set of measurements for *one* sample. These are all the measurements in all configurations for that one sample. 

### Repetition
A repetition is a single measurement in a single configuration. It contains the measurement and the preceding direct beam and direct-beam-through-sample measurements, from which the primary beam flux, beam position and transmission factor is determined. 

## expected directory structure
The directory structure is expected to be organized as:

```bash
├─── Proposals
└─── Measurements
    ├─── SAXS002
    │   ├─── logbooks
    │   └─── data
    │       └─── Masks
    │       └─── 2025
    │           └─── 20250101  # (measurement date)
    │               └─── 20250101_[batch]_[repetition] # directory with files
    │                   └───eiger_[number]_master.h5
    │                   └───eiger_[number]_data00001.h5
    │                   └───im_craw.nxs
    │                   └─── beam_profile
    │                       └─── eiger_[number]_master.h5
    │                       └─── eiger_[number]_data00001.h5
    │                       └─── im_craw.nxs
    │                   └───beam_profile_through_sample
    │                       └─── eiger_[number]_master.h5
    │                       └─── eiger_[number]_data00001.h5
    │                       └─── im_craw.nxs
    │               └─── 20250101_[batch]_[repetition]
    │               └─── ...
    │               └─── autoproc  # (processed datafiles)    
```

# usage example:  

```zsh
python src/directory_processor.py --config MOUSE_settings.yaml --single_dir ~/Documents/BAM/Measurements/newMouseTest/Measurements/SAXS002/data/2025/20250101/20250101_21_22  --steps processstep_translator_step_1 processstep_translator_step_2 processstep_beamanalysis
```
or 
```zsh
python src/directory_processor.py --config MOUSE_settings.yaml --ymd 20250101 --batch 21 --repetition 22 --steps processstep_translator_step_1 processstep_translator_step_2 processstep_beamanalysis
```

# top-level methods: 

## 1. `process_all_data`
This processes all data for a given measurment date [YYYYMMDD]. This is also available as a shell script

## 2. `process_single_data`
This processes one single set of datafiles. 

## 3. `watcher`
This watches a measurement date and preliminarily processes the files in complete repetitions as they appear.

## 4. `check_source_files`
This runs processing_possible for all directories. all missing files are listed. 

# functionality methods:

## `checkers`
This set of methods contains
  1. `processing_possible`: this checks if all the required files are present. 
  2. `already_processed`: this checks if there is already a processed output file in place. Only used for the watcher

