#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP iterator"
echo "-----------------------"
echo ""

echo "Parameters for all networks:"
echo "  Input:          [pode, Z,   H,   PV]    "
set=3
echo "  Size:           [64, 64, 64]            "
echo "  n_epochs        100                     "
epochs=100
echo "  Training run:   000 (--> only PODE3)    "
name_train=000
echo ""

echo "Network 001: Labels [T, P]"
name_net=001
python MLP_temperature.py -mech cai -nbr_run $name_train --n_epochs $epochs -nbr_net $name_net --feature_set $set --hidden 64 64 64 --labels T P

echo ""
echo ""

echo "Network 002: Labels [CO2, H2O, CO]"
name_net=002
python MLP_temperature.py -mech cai -nbr_run $name_train --n_epochs $epochs -nbr_net $name_net --feature_set $set --hidden 64 64 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 003: Labels [PODE, O2]"
name_net=003
python MLP_temperature.py -mech cai -nbr_run $name_train --n_epochs $epochs -nbr_net $name_net --feature_set $set --hidden 64 64 64 --labels O2 PODE

echo ""
echo ""

echo "Network 004: Labels [Q]"
name_net=004
python MLP_temperature.py -mech cai -nbr_run $name_train --n_epochs $epochs -nbr_net $name_net --feature_set $set --hidden 64 64 64 --labels Q
