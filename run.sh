#!/bin/bash

# This script is used to run the entire pipeline of inference and evaluation for the specified datasets, models, and methods. It first calculates the uncertainty of the model's predictions using "uncertainty.py" and then runs the inference and evaluation using "inference.py" and "evaluation.py".

methods=("finalActUIE")  # Example method, modify as needed
datasets=("CoNLL03") # ace04NER conll2003 conll2004 SciERC
models=("deepseek-chat") # gemma3:14b

home="/home/zd/work/APIE"

today=$(date "+%m-%d-%H")
client="deepseek-api"  # Example client, modify as needed

UNshotSize=2
UNresponseSize=3

alpha=0.33
beta=0.33
gama=0.33

shotSize=3

for dataset in $datasets; do
    for model in $models; do
        for method in $methods; do

            echo "Running Inference: $dataset with $model using $method"
            python inference.py \
                --inputFile $home/data/APIE/$dataset/test.json \
                --schema $home/data/APIE/$dataset/schema.json \
                --outputFile $home/output/$dataset/$model/$method-$today.json \
                --model $model \
                --client $client \
                --pollSize -1 \
                --shotSize $shotSize \
                --method $method \
                --uncertaintyFile $home/data/APIE/$dataset/$model-uncertainty-$UNshotSize-$UNresponseSize.json \
                --frontArgs 0.05 \
                --alpha $alpha \
                --beta $beta \
                --gama $gama

            echo "Running Evaluation"
            python evaluation.py \
                --inputFile $home/output/$dataset/$model/$method-$today.json \
                --outputFile $home/output/$dataset/$model/$method-$today.json \
                --schema $home/data/APIE/$dataset/schema.json \
                --shotSize $shotSize \
                --recordFile $home/output/$dataset/$model/$method-record.json \
                --uncertaintyFile $home/output/$today/$dataset/$model-uncertainty-$UNshotSize-$UNresponseSize.json \
                --alpha $alpha \
                --beta $beta \
                --gama $gama

        done
    done
done