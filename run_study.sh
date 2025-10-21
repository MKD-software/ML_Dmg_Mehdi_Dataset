#!/bin/bash

# Remove old script
rm -f plot_explore_data.py

# Convert notebook to Python script with desired name
jupyter nbconvert --to script plot_explore_data.ipynb --output run_study

# Run the script
python3 run_study.py
