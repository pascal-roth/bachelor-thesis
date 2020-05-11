#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All training data used in the BA of Pascal Roth will be created:"
echo "----------------------------------------------------------------"
echo ""

echo "Cai Mechanism with parameter setting: Phi = [0.5, 1.0, 1.5], P = [10, 20, 40]bar for a temperature range (650, 1250,15)"
python iterator.py -mech cai -phi 0.0 -p 0 --pode 3 -inf_print False
python iterator.py -mech cai -phi 0.0 -p 0 --pode 4 -inf_print False

