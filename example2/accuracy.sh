#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

base_dir="~/sharenet/example2"
results_dir="${base_dir}/data"

file_name="pidc.edges.txt.gz"
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "STRING" 
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "nonspecific_chip" 
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "specific_chip" 

base_dir="/data/cb/alexwu/sharenet/example2"
results_dir="${base_dir}/results"

file_name="sharenet.nc10.thresh1.0.pidc.edges.txt"
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "STRING" 
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "nonspecific_chip" 
python -u ~/sharenet/sharenet_accuracy.py -d $base_dir -r $results_dir -K 24 -f $file_name -rn "specific_chip" 

