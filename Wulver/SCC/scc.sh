#!/bin/bash -l
#SBATCH --job-name=SCC
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=zd4 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-40:59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=60000M
python SCC.py ../connectome_graph.csv 
