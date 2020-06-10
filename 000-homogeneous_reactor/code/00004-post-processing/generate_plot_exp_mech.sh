#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All exp-mech plots in the BA of Pascal Roth will be created:"
echo "---------------------------------------------------"
echo "Beforehand the generate_data_exp.sh script has to be executed"
echo ""

echo "Experimental Data taken from the paper of Jacobs et al:"
python plot_exp_mech.py -phi 1.0 -p 20 --pode 1 -nbr_run 000
python plot_exp_mech.py -phi 1.0 -p 20 --pode 2 -nbr_run 000
python plot_exp_mech.py -phi 1.0 -p 20 --pode 3 -nbr_run 000

python plot_exp_mech.py -phi 1.0 -p 10 --pode 3 -nbr_run 000
python plot_exp_mech.py -phi 1.0 -p 10 --pode 4 -nbr_run 000

echo "Experimental Data taken from the paper of He et al:"
python plot_exp_mech.py -phi 0.5 -p 10 --pode 3 -nbr_run 001
python plot_exp_mech.py -phi 0.5 -p 15 --pode 3 -nbr_run 001

python plot_exp_mech.py -phi 1.0 -p 10 --pode 3 -nbr_run 002
python plot_exp_mech.py -phi 1.0 -p 15 --pode 3 -nbr_run 002

python plot_exp_mech.py -phi 1.5 -p 10 --pode 3 -nbr_run 003
python plot_exp_mech.py -phi 1.5 -p 15 --pode 3 -nbr_run 003
