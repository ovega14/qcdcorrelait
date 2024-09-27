#!/bin/bash
set -e
export OMP_NUM_THREADS=1
#ulimit -Sv 9000000
seed=42
num=0

mo1label=("0.164")
mo2label=("0.01555")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.00311" "0.00311" "0.00311" "0.00311" "0.00311" "0.00311")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.01555" "0.01555" "0.01555" "0.01555" "0.01555" "0.01555")

params=()
for i in "${!mo1label[@]}"
do
    param=("P5-P5_RW_RW_d_d_m"${mo1label[$i]}"_m"${mo2label[$i]}"_p000")
    read output_dataname <<< "$param"
    echo TRUTH_$output_dataname
    NOW=$(date +"%Y-%m-%d-%H-%M-%S")
    num=$num
    name=fit_TRUTH_mo1_${mo1label[$i]}_mo2_${mo2label[$i]}+$NOW+$num
    if [ -d "results/"$name ]
    then
        echo "results/"$name" already exists"
        read -n 1 -p "keep running? [y/n]" prompt
        if [[ $prompt == "y" || $prompt == "yes" || $prompt == "Yes" ]]
        then
            echo "proceed"
        else
            echo -e "\nexit"
            exit
        fi
    fi
    mkdir -p ../results/$name
    mkdir ../results/$name/data
    mkdir ../results/$name/plots
    python -W ignore ../frontend/fit_truth.py \
        --seed $seed \
        --output_dataname $output_dataname \
        --results_dir "../results/$name"
    rm ../results/$name/*.pkl
done