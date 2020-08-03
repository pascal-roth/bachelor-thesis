#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP iterator"
echo "-----------------------"
echo ""

echo "Parameters for all networks:"
echo "  Input:          [pode, Z,   H,   PV]    "
echo "  For pressure    20 bar                  "
set=3
pressure=20
echo "  n_epochs        100                     "
epochs=100
echo "  Training run:   000 (--> only PODE3)    "
name_train=000
echo "  device:         gpu_multi               "
device=gpu_multi
echo "  batch number:   100                     "
batches=100
echo ""

echo "Network 000: Labels [T, P], Size: [64, 64, 64]"
name_net=000
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 64 64 --labels T P

echo ""
echo ""

echo "Network 001: Labels [CO2, H2O, CO], Size: [64, 64, 64]"
name_net=001
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 64 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 002: Labels [PODE, O2], Size: [64, 64, 64]"
name_net=002
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 64 64 --labels O2 PODE

echo ""
echo ""

echo "Network 003: Labels [HRR], Size: [64, 64, 64]"
name_net=003
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 64 64 --labels HRR

echo ""
echo ""

echo "Network 004: Labels [T, P], Size: [64, 64, 64]"
name_net=004
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 64 --labels T P

echo ""
echo ""

echo "Network 005: Labels [CO2, H2O, CO], Size: [64, 64, 64]"
name_net=005
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 006: Labels [PODE, O2], Size: [64, 64, 64]"
name_net=006
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 007: Labels [HRR], Size: [64, 128, 64]"
name_net=007
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 64 --labels HRR

echo ""
echo "All MLPs are expanded due to new insights gained from the previous MLPs"
echo ""

echo "Parameters for all networks:"
echo "  Input:          [pode, Z,   H,   PV]    "
echo "  For pressure    20 bar                  "
set=3
pressure=20
echo "  Training run:   000 (--> only PODE3)    "
name_train=000
echo "  device:         gpu_multi               "
device=gpu_multi
echo "  batch number:   100                     "
batches=100
echo ""

echo "Network 008: Labels [T, P], Size: [64, 128, 128, 64]"
name_net=008
epochs=200
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
epochs=200
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 011: Labels [HRR], Size: [64, 128, 256, 128, 64]"
name_net=011
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels HRR

echo ""
echo ""

echo "Network 012: Labels [T, P], Size: [64, 128, 256, 128 64]"
name_net=012
epochs=200
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels T P

echo ""
echo ""

echo "Network 013: Labels [CO2, H2O, CO], Size: [64, 128, 256, 256, 128 64]"
name_net=013
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 256 128 64 --labels CO2 CO H2O

echo ""
echo ""

echo "Network 014: Labels [PODE, O2], Size: [64, 128, 256, 128 64]"
name_net=014
epochs=200
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 128 64 --labels O2 PODE

echo ""
echo ""

echo "Network 015: Labels [HRR], Size: [64, 128, 256, 256, 128 64]"
name_net=015
epochs=300
python MLP_temperature.py -mech cai -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --device $device -b_frac $batches --hidden 64 128 256 256 128 64 --labels HRR

exi
