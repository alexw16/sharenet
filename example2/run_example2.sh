#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

data_dir="~/sharenet/example2/data"
results_dir="~/sharenet/example2/results"
script_dir="~/sharenet"

python -u "${script_dir}/sharenet_example2.py" -d $data_dir -r $results_dir -f "pidc.edges.txt.gz" -sf "pidc.edges.txt.gz" -K 24 -nc 10 -thr 1 -tol 0.01 
