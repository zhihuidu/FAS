#!/bin/bash -l
#SBATCH --job-name=DFS
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=bigmem
#SBATCH --qos=standard
#SBATCH --account=zd4 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-40:59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=250000M
python DFS.py tmp.csv >>output.my.txt
