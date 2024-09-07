#!/bin/bash
#SBATCH --job-name=m-heavy  # Job name
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=128         # Number of CPU cores per task
#SBATCH --mem=512G                  # Memory per node (e.g., 4 GB)
#SBATCH --time=90000:30:00           # Time limit (hh:mm:ss)

# Run your job command (e.g., Python script)
python -u MulHeavyMaxFlow.py ../../../graph.csv |tee smallnumberofvertex2-output.txt
