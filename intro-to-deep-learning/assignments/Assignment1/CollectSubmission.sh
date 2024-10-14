#!/bin/bash

if [ -z "$1" ]; then
    echo "student ID is required.
Usage: ./CollectSubmission 20xx_xxxxx"
    exit 0
fi

files="Assignment1-1_MLP.ipynb
Assignment1-2_CNN.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done


rm -f $1.tar.gz
mkdir $1
cp -r ./*.ipynb $1/
tar cvzf $1.tar.gz $1
rm -rf $1