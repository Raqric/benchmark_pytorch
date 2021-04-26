#!/bin/bash
#SBATCH -p cola-corta  #default queue allows only one node
#SBATCH -t 00:20:00 #(min of execution time)
#SBATCH -c 1 #(24 cores per job)

export OMP_NUM_THREADS=1
echo "Execution with 1 thread"
python3 benchmark-cpu-kaggle.py 

export OMP_NUM_THREADS=2
echo "Execution with 2 thread"
python3 benchmark-cpu-kaggle.py

export OMP_NUM_THREADS=4
echo "Execution with 4 thread"
python3 benchmark-cpu-kaggle.py

export OMP_NUM_THREADS=8
echo "Execution with 8 thread"
python3 benchmark-cpu-kaggle.py

export OMP_NUM_THREADS=16
echo "Execution with 16 thread"
python3 benchmark-cpu-kaggle.py


export OMP_NUM_THREADS=24
echo "Execution with 24 thread"
python3 benchmark-cpu-kaggle.py




