#!/bin/bash

if [ -z "$1" ]; then
    echo "student ID is required.
Usage: ./CollectSubmission 20xx_xxxxx"
    exit 0
fi

files="Assignment2-1_RNN.ipynb
Assignment2-2_Transformers.ipynb"

for file in $files
do
    if [ ! -f "$file" ]; then
        echo "Required $file not found."
        exit 0
    fi
done

if [ ! -d "model_checkpoints" ]; then
    echo "Required model_checkpoints directory not found."
    exit 0
fi

rm -f "$1.tar.gz"
mkdir "$1"
cp -r ./*.ipynb "$1/"
cp -r ./model_checkpoints "$1/"
tar cvzf "$1.tar.gz" "$1"
rm -rf "$1"