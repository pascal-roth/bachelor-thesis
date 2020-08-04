#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP iterator"
echo "-----------------------"
echo "All MLPs are expanded due to new insights gained from the MLPs of MLP_iter_1"
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

echo "Network 000: Labels [T, P], Size: [64, 128, 128, 64]"
name_net=000
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 128 64 --labels T P

echo ""
echo ""

echo "Network 001: Labels [CO2, H2O, CO], Size: [64, 128, 256, 128, 64]"
name_net=001
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 002: Labels [PODE, O2], Size: [64, 128, 128, 64]"
name_net=002
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 003: Labels [HRR], Size: [64, 128, 256, 128, 64]"
name_net=003
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels HRR

echo ""
echo ""

echo "Network 004: Labels [T, P], Size: [64, 128, 256, 128 64]"
name_net=004
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels T P

echo ""
echo ""

echo "Network 005: Labels [CO2, H2O, CO], Size: [64, 128, 256, 256, 128 64]"
name_net=005
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 256 128 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 006: Labels [PODE, O2], Size: [64, 128, 256, 128 64]"
name_net=006
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 007: Labels [HRR], Size: [64, 128, 256, 256, 128 64]"
name_net=007
epochs=150
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 256 128 64 --labels HRR

exit
