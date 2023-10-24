#!/bin/bash

# Request cores
#SBATCH --cpus-per-task=20

# Request memory per thread
#SBATCH --mem-per-cpu=3G

# Set a job name
#SBATCH --job-name=GW14

# Specify time before the job is killed by scheduler (in case is hangs). In this case - 24 hours
#SBATCH --time=24:00:00

# Declare the merged STDOUT/STDERR file
#SBATCH --output=/data/Graph4Patients/stdout/output_%j.out

# Run something
cd ~/HeartNet
python3 -m src.heartnet.heartnet --config ~/configs/config14.toml
