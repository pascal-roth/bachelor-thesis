#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP temperature"
echo "--------------------------"
echo ""

echo "Which data should be used for the MLP"
read -p "mechanism         [he, sun, cai]:                 " mechanism
read -p "name of train run [XXX]:                          " name_train
read -p "name of test run  [XXX]:                          " name_test

while true; do
	echo ""
	read -p "Exclude data from training/testing (ar = array entry possible)?" yn2
	echo ""
	case $yn2 in
		[Yy]* ) read -p "PODE_n            [1, 2, 3, 4]    (ar): " pode_n
                read -p "equivalence ratio [0.5, 1.0, 1.5] (ar): " equivalence_ratio
                read -p "pressure          [10, 20, 40]    (ar): " pressure
                read -p "temperature       (650, 1250, 15) (ar): " temp; break;;
		[Nn]* ) pode_n=0
		        equivalence_ratio=0
		        pressure=0
		        temp=0; break;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

while true; do
	echo ""
	read -p "Should a pre-trained model be load?" yn1
	echo ""
	case $yn1 in
		[Yy]* ) echo "Decide training and network parameters"
                read -p "nbr of train epochs:                              " epochs
                read -p "nbr of network:                                   " name_net
		        read -p "Which validation method model  [train, test]:     " typ
		        python MLP_temperature.py -mech $mechanism -nbr_run $name_train -nbr_test $name_test --pode $pode_n -phi $equivalence_ratio -p $pressure -temp $temp --n_epochs $epochs -nbr_net $name_net --typ $typ ; break;;
		[Nn]* ) echo "Decide NN building and training parameters"
                read -p "Input parameters  [pode, phi, P_0, T_0, PV] (ar): " samples
                read -p "Output parameters [T]                       (ar): " labels
                read -p "nbr of train epochs:                              " epochs
                read -p "nbr of network:                                   " name_net
		        typ=train
		        python MLP_temperature.py -mech $mechanism -nbr_run $name_train -nbr_test $name_test --pode $pode_n -phi $equivalence_ratio -p $pressure -temp $temp -s_paras $samples -l_paras $labels --n_epochs $epochs -nbr_net $name_net; exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

