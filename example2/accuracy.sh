#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

script_dir=".." 
base_dir="${script_dir}/example2"
results_dir="${base_dir}/data"

file_name="pidc.edges.txt.gz"
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "STRING" 
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "nonspecific_chip" 
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "specific_chip" 

script_dir=".."
base_dir="${script_dir}/example2"
results_dir="${base_dir}/results"

file_name="sharenet.nc10.pidc.edges.txt"
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "STRING" 
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "nonspecific_chip" 
python -u "${script_dir}/sharenet_accuracy.py" -d $base_dir -r $results_dir -K 24 -f $file_name -rn "specific_chip" 

