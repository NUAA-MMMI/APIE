#!/bin/bash

datasets=("ace04NER conll2004 SciERC")
models=("qwen2.5:14b")
shotSize=2
responseSize=5

home="/home/zd/work/Act-UIE"
client="ollama"

for dataset in $datasets; do
    for model in $models; do
        echo "Running on dataset: $dataset with model: $model"
        echo "Running Uncertainty Calculation"
        python uncertainty.py \
            --model $model \
            --client $client \
            --inputFile $home/data/Act-UIE/$dataset/test.json \
            --schema $home/data/Act-UIE/$dataset/schema.json \
            --shotSize $shotSize \
            --responseSize $responseSize \
            --pollSize -1 \
            --uncertaintyFile /home/zd/work/Act-UIE/data/Act-UIE/$dataset/$model-uncertainty-$shotSize-$responseSize.json\
            --method ActUIE
    done
done