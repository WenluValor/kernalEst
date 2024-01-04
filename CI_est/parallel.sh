#!/bin/bash

# Define the function to run
run_regress() {
    python rep_boots.py "$1"
}

# Export the function so it's available to parallel
export -f run_regress

# Run regress function with parameter -1
run_regress -1

# Run regress function in parallel for s ranging from 0 to 198
# seq 0 20 | parallel -j "$(nproc)" run_regress
NUM_CORES=$(getconf _NPROCESSORS_ONLN)
RUN_CORES=$((NUM_CORES/2))
seq 0 5 | parallel -j "$RUN_CORES" run_regress