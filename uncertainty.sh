#!/bin/bash

# This script is used to calculate the uncertainty of the model's predictions on the test set of the specified dataset.

# datasets name
datasets=("CoNLL03")
# models name
models=("deepseek-chat")
# number of shots and responses for uncertainty calculation
shotSize=2
responseSize=3

# change is path according to your local environment
home="/home/zd/work/APIE"
# client is used to specify the API client for uncertainty calculation, which is "deepseek-api" in this case. You can change it to other clients if needed,such as "ollama" in "uncertainty.py".
client="deepseek-api"


for dataset in $datasets; do
    for model in $models; do
        echo "Running on dataset: $dataset with model: $model"
        echo "Running Uncertainty Calculation"
        python uncertainty.py \
            --model $model \
            --client $client \
            --inputFile $home/data/APIE/$dataset/test.json \
            --schema $home/data/APIE/$dataset/schema.json \
            --shotSize $shotSize \
            --responseSize $responseSize \
            --pollSize -2 \
            --uncertaintyFile $home/data/APIE/$dataset/$model-uncertainty-$shotSize-$responseSize.json\
            --method ActUIE
    done
done