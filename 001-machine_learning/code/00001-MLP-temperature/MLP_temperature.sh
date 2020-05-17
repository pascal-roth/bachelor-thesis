#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP temperature"
echo "--------------------------"
echo ""

echo "Which data should be used for the MLP"
read -p "mechanism         [he, sun, cai]:       " mechanism
read -p "name of train run [XXX]:                " name_train
read -p "name of test run  [XXX]:                " name_test
echo ""
echo "Decide NN building and training parameters"
read -p "nbr of train epochs:                    " epochs
read -p "nbr of network:                         " name_net

while true; do
	echo ""
	read -p "Exclude data from training (ar = array entry possible)?" yn
	echo ""
	case $yn in
		[Yy]* ) read -p "PODE_n            [1, 2, 3, 4]    (ar): " pode_n
                read -p "equivalence ratio [0.5, 1.0, 1.5] (ar): " equivalence_ratio
                read -p "pressure          [10, 20, 40]    (ar): " pressure
                read -p "temperature       (650, 1250, 15) (ar): " temp
			    python MLP_temperature.py -mech $mechanism -nbr_run $name_train -nbr_test $name_test --n_epochs $epochs -nbr_net $name_net -phi $equivalence_ratio -p $pressure --pode $pode_n -temp $temp; break;;
		[Nn]* ) python MLP_temperature.py -mech $mechanism -nbr_run $name_train -nbr_test $name_test --n_epochs $epochs -nbr_net $name_net; exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done