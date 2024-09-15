#!/bin/bash -l
#SBATCH --job-name=lp
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
##SBATCH --partition=bigmem
##SBATCH --qos=standard
##SBATCH --account=zd4 # Replace PI_ucid which the NJIT UCID of PI
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
##SBATCH --time=2-23:59:00  # D-HH:MM:SS
##SBATCH --mem-per-cpu=250000M
python -u lp.py ../../../../graph.csv
