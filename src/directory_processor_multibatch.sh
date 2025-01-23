#!/bin/bash

# Ensure the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <YMD> <min_batch> <max_batch>"
    exit 1
fi

# Assign the input arguments to variables
YMD="$1"
MIN_BATCH="$2"
MAX_BATCH="$3"

# Iterate over the batch numbers within the specified range
for ((batch=MIN_BATCH; batch<=MAX_BATCH; batch++)); do
    echo "Processing batch $batch for YMD $YMD"
    
    python src/directory_processor.py --config MOUSE_settings.yaml --ymd "$YMD" \
    --batch "$batch" --parallel --steps \
    processstep_translator_step_1 \
    processstep_translator_step_2 \
    processstep_beamanalysis \
    processstep_cleanup_files \
    processstep_add_mask_file \
    processstep_metadata_update \
    processstep_add_background_files \
    processstep_thickness_from_absorption \
    processstep_stacker

    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to process batch $batch. Exiting."
        # exit 1
    fi

    echo "Completed batch $batch"
done

echo "All batches processed successfully."