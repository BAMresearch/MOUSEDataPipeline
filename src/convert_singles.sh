#!/bin/zsh
# This script applies the HDF5 Translation steps to all files in a tree defined by measurement date
# For testing.. normally this would be /mnt/vsi-db/
base_dir="/Users/bpauw/Documents/BAM/Measurements/newMouseTest/Measurements/SAXS002"
hdf5translator_dir="/Users/bpauw/Code/HDF5Translator/"
measurement_YMD=$1
max_parallel=${2:-4}  # Default to 4 parallel processes if $2 is not provided
measurement_year="${measurement_YMD:0:4}"

echo "Processing measurements from $measurement_YMD, year $measurement_year"

# Load virtual environment
source "$hdf5translator_dir/.venv/bin/activate"

# Go to base directory
cd "$base_dir" || exit

# Build YAML files
python "$hdf5translator_dir/src/tools/excel_translator.py" -I "data/TranslatorConfigurations/BAM_new_MOUSE_xenocs_translator_configuration.xlsx"
python "$hdf5translator_dir/src/tools/excel_translator.py" -I "data/TranslatorConfigurations/BAM_new_MOUSE_dectris_adder_configuration.xlsx"

# Process each file sequentially
find "data/$measurement_year/$measurement_YMD" -type f -depth 2 -name "im_craw.nxs" -print0 | while IFS= read -r -d '' file; do
    # echo "Processing $file"
    # Get the directory of the file and its basename
    bpath=$(dirname "$file")
    bname=$(basename "$bpath")
    
    # Find the eiger file that matches the pattern
    eiger_file=$(find "$bpath" -name "eiger_*_master.h5" -print -quit)
    echo "Processing $file" #, bpath: $bpath, bname: $bname Eiger file: $eiger_file"
    if [ -z "$eiger_file" ]; then
        echo "No eiger file found for $bpath"
        continue
    fi

    # Run the first HDF5 Translator command
    python3 -m HDF5Translator \
        -C "data/TranslatorConfigurations/BAM_new_MOUSE_xenocs_translator_configuration.yaml" \
        -I "$bpath/im_craw.nxs" \
        -O "$bpath/$bname.nxs" \
        -d && \
    # Run the second HDF5 Translator command
    python3 -m HDF5Translator \
        -C "data/TranslatorConfigurations/BAM_new_MOUSE_dectris_adder_configuration.yaml" \
        -I "$eiger_file" \
        -T "$bpath/$bname.nxs" \
        -O "$bpath/MOUSE_$bname.nxs" \
        -d \\ && \
    echo "Finished processing $file"
done
