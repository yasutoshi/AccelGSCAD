#!/bin/bash
# select method
methods=("group_scad" "skip_group_scad" "fast_group_scad")

# select dataset
datasets=("eunite" "qsbralks")
#datasets=("qsbr_rw1" "qsf" "triazines")

# select lambda_factor
lambda_factors=("0.001" "0.0001")

for i in "${methods[@]}"
do
    for j in "${datasets[@]}" 
    do
        for k in "${lambda_factors[@]}"
        do
            python3 experiment.py --method ${i} --data ${j} --lambda_factor ${k} > log_${j}_${k}_${i}.txt
        done
    done
done
