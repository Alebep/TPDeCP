#!/bin/bash
#SBATCH --time=1:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20

time ./cuda_k_means 10000000 32 $THREADS