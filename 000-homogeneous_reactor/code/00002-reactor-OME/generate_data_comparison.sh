#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Data for the comparison of the MLP-model and GRM-model will be generated:"
echo "-------------------------------------------------------------------------"
echo "Includes data to train MLPs with the different PV and data necessary for the comparison"
echo ""

while true; do
	echo ""
	read -p "Create Training Data for the MLP model?" yn
	echo ""
	case $yn in
		[Yy]* ) echo "Cai Mechanism with Phi=(0.5, 0.67, 1.0), P=40, T=[680, 1240, 40]"
		        echo "Saved in train run nbr 002"
            python iterator_multi.py -mech cai --pode 3 4 -nbr_run 002 --category train --phi_0 0.5 --phi_end 1.5 --phi_step 0.05 --p_0 40 --p_end 40 --p_step 40 -t_0  650 -t_end  1250 -t_step 30 -comp; break;;
		[Nn]* ) break;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

while true; do
	echo ""
	read -p "Create Test Data for the MLP model?" yn2
	echo ""
	case $yn2 in
		[Yy]* ) echo "Cai Mechanism with Phi=(0.5, 0.67, 1.0), P=40, T=[680, 1240, 40]"
		        echo "Saved in test run nbr 002"
            python iterator_multi.py -mech cai --pode 3 4 -nbr_run 002 --category test --phi_0 0.5 --phi_end 1.0 --phi_step 0.5 --p_0 40 --p_end 40 --p_step 40 -t_0  680 -t_end  1240 -t_step 40 -comp
            python iterator_multi.py -mech cai --pode 3 4 -nbr_run 002 --category test --phi_0 0.67 --phi_end 0.67 --phi_step 0.67 --p_0 40 --p_end 40 --p_step 40 -t_0  680 -t_end  1240 -t_step 40 -comp; break;;
		[Nn]* ) break;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

exit