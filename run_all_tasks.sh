#!/bin/bash

# Run all tasks consecutively
# This script runs all the tasks in the repository in sequence

echo "Starting all tasks..."


echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
# Task 01: Model Training Sweep
echo "Running Task 01: Model Training Sweep..."
python runs/train.py +experiment=sweep_models --multirun

printf "\n%.0s" {1..5}
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
# Task 02: Generate Report
echo "Running Task 02: Generate Report..."
python runs/report.py

echo "All tasks completed!" 