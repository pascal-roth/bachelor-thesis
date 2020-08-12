#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP iterator"
echo "-----------------------"
echo "Architecture of MLPs same as MLP_iter_2, but lr scheduler works with valid_loss"
echo ""

echo "Parameters for all networks:"
echo "  Input:          [pode, Z,   H,   PV]    "
echo "  For pressure    40 bar                  "
set=3
pressure=40
echo "  Training run:   001 (PODE3 and PODE4)    "
name_train=001
echo "  device:         gpu_multi               "
device=gpu_multi
echo "  batch number:   100                     "
batches=100
echo ""

echo "Network 008: Labels [T, P], Size: [64, 128, 128, 64]"
name_net=008
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 128 64 --labels T P

echo ""
echo ""

echo "Network 009: Labels [CO2, H2O, CO], Size: [64, 128, 256, 128, 64]"
name_net=009
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 010: Labels [PODE, O2], Size: [64, 128, 128, 64]"
name_net=010
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 011: Labels [HRR], Size: [64, 128, 256, 128, 64]"
name_net=011
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels HRR

echo ""
echo ""
