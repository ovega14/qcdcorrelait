#!/bin/bash
export OMP_NUM_THREADS=1
#ulimit -Sv 9000000
seed=42
use_torch=1
num=0
#dict_hyperparams='{"lr":0.01,"l2_coeff":1e-2,"training_steps":500}'
rel_eps=1e-2
# respecify params here !!!
reg_methods=("LinearModel" "MLP" "CNN" "Transformer")
reg_method0="CNN"

mi1label=("0.548")
mi2label=("0.01555")
mo1label=("0.164")
mo2label=("0.01555")
# mi1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mi2label=("0.01555" "0.01555" "0.01555" "0.01555" "0.01555" "0.01555")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.00311" "0.00311" "0.00311" "0.00311" "0.00311" "0.00311")
# mi1label=("0.164" "0.164" "0.164" "0.164" "0.164" "0.164")
# mi2label=("0.01555" "0.01555" "0.01555" "0.01555" "0.01555" "0.01555")
# mo1label=("0.164" "0.1827" "0.365" "0.548" "0.731" "0.843")
# mo2label=("0.01555" "0.01555" "0.01555" "0.01555" "0.01555" "0.01555")

params=()
for i in "${!mi1label[@]}"
do
    param=("$use_torch $reg_method0 P5-P5_RW_RW_d_d_m"${mi1label[$i]}"_m"${mi2label[$i]}"_p000 P5-P5_RW_RW_d_d_m"${mo1label[$i]}"_m"${mo2label[$i]}"_p000 {\"lr\":0.01,\"l2_coeff\":1e-2,\"training_steps\":200}")
    read use_torch reg_method input_dataname output_dataname dict_hyperparams<<< "$param"
    echo $use_torch $reg_method $input_dataname $output_dataname $dict_hyperparams
    NOW=$(date +"%Y-%m-%d-%H-%M-%S")
    num=$num
    name=$reg_method+mi1_${mi1label[$i]}_mi2_${mi2label[$i]}_mo1_${mo1label[$i]}_mo2_${mo2label[$i]}+$NOW+$num
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
    #cp *.py results/$name/
    #cp *.sh results/$name/
    # cp -r record results/$name/
    #cd ../results/$name
    mkdir ../results/$name/data
    mkdir ../results/$name/plots
    mkdir ../results/$name/model
    mkdir ../results/$name/results
		python -W ignore test_model.py \
        --seed $seed \
        --use_torch $use_torch \
        --input_dataname $input_dataname \
        --output_dataname $output_dataname \
        --reg_method $reg_method \
        --dict_hyperparams $dict_hyperparams \
        --rel_eps $rel_eps \
        --results_dir "../results/$name"
    #cd ../../
done