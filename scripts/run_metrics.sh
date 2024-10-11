#!/bin/bash
export OMP_NUM_THREADS=1
#ulimit -Sv 9000000
seed=42
num=0

lr=0.01
l2_coeff=1e-2
training_steps=200

mi1label=("0.548")
mi2label=("0.01555")
mo1label=("0.164")
mo2label=("0.01555")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.00311" "0.00311" "0.00311" "0.00311" "0.00311" "0.00311")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.01555" "0.01555" "0.01555" "0.01555" "0.01555" "0.01555")

params=()
for i in "${!mi1label[@]}"
do
    param=("P5-P5_RW_RW_d_d_m"${mi1label[$i]}"_m"${mi2label[$i]}"_p000 P5-P5_RW_RW_d_d_m"${mo1label[$i]}"_m"${mo2label[$i]}"_p000 {\"lr\":0.01,\"l2_coeff\":1e-2,\"training_steps\":200}")
    read input_dataname output_dataname dict_hyperparams<<< "$param"
    echo $input_dataname $output_dataname $dict_hyperparams
    NOW=$(date +"%Y-%m-%d-%H-%M-%S")
    num=$num
    name=RUN_METRICS+mi1_${mi1label[$i]}_mi2_${mi2label[$i]}_mo1_${mo1label[$i]}_mo2_${mo2label[$i]}+$NOW+$num
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
        --lr $lr \
        --l2_coeff $l2_coeff \
        --training_steps $training_steps \
        --input_dataname $input_dataname \
        --output_dataname $output_dataname \
        --results_dir "../results/$name"
    #rm ../results/$name/*.pkl
done