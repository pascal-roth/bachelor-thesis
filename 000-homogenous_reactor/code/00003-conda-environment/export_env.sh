#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

conda env export > environment.yml

echo "Conda environment created"