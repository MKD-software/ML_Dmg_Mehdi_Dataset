#!/bin/bash

# Use provided notebook name or default to plot_explore_data
NOTEBOOK="${1:-plot_explore_data}.ipynb"
OUTPUT="run_study"

# Remove old script
rm -f "${OUTPUT}.py"

# Convert notebook to Python script with desired name
jupyter nbconvert --to script "src/${NOTEBOOK}" --output "${OUTPUT}"

# Run the script
python3 "${OUTPUT}.py"
