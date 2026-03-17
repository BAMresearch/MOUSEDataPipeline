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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Iterate over the batch numbers within the specified range

"$SCRIPT_DIR/directory_processor_multibatch_nostack.sh" "$YMD" "$MIN_BATCH" "$MAX_BATCH"
"$SCRIPT_DIR/directory_processor_multibatch_stackonly.sh" "$YMD" "$MIN_BATCH" "$MAX_BATCH"

echo "All batches processed successfully."
