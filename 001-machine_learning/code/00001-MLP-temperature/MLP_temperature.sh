#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP temperature"
echo "--------------------------"
echo ""

echo "Which data should be used for the MLP"
read -p "mechanism         [he, sun, cai]:                 " mechanism
read -p "name of train run [XXX]:                          " name_train

echo ""

echo "Which device should be used for training?"
read -p "device            [cpu, gpu, gpu_multi]:          " device

while true; do
	echo ""
	read -p "Should a pre-trained model be load?" yn1
	echo ""
	case $yn1 in
		[Yy]* ) echo "Decide training and network parameters"
                read -p "nbr of train epochs:                              " epochs
                read -p "nbr of network:                                   " name_net
		        python MLP_temperature.py -mech $mechanism -nbr_run $name_train --n_epochs $epochs -nbr_net $name_net --device $device; break;;
		[Nn]* ) echo "Decide NN building and training parameters"
                read -p "nbr of network:                                   " name_net
                read -p "nbr of train epochs:                              " epochs
                read -p "size of the NN (neurons for hidden layer)   (ar): " hidden
                read -p "Output parameters [T]                       (ar): " labels
                echo "Decide which training feature set"
                echo "  1) [pode, phi, P_0, T_0, PV]"
                echo "  2) [pode, Z,   P,   H,   PV]"
                echo "  3) [pode, Z,   H,   PV     ]"
                echo "  4) [pode, Z,   P_0, H,   PV]"
                read -p "Chose set (1, 2, 3, 4):                           " set
                case $set in
		            [3]*  ) read -p "select pressure of network   (10, 40, 2)    (ar): " pressure
                            python MLP_temperature.py -mech $mechanism -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --labels $labels --hidden $hidden --device $device; break;;
		            * )     python MLP_temperature.py -mech $mechanism -nbr_run $name_train -p $pressure --n_epochs $epochs -nbr_net $name_net --feature_set $set --labels $labels --hidden $hidden --device $device; exit;;
	            esac
		        exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

