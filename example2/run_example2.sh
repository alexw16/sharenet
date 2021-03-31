#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

script_dir=".."
data_dir="${script_dir}/example2/data"
results_dir="${script_dir}/example2/results"

python -u "${script_dir}/sharenet_example2.py" -d $data_dir -r $results_dir -f "pidc.edges.txt.gz" -sf "pidc.edges.txt.gz" -K 24 -nc 10 -tol 0.01 
