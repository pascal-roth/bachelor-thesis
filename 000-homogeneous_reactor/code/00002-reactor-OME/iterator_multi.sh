#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to data iterator"
echo "------------------------"
echo ""

echo "Enter parameters of the homogeneous reactor (ar = array entry possible)"
read -p "mechanism         [he, sun, cai]:       " mechanism
read -p "PODE_n            [1, 2, 3, 4]    (ar): " pode_n
read -p "name of run       [XXX]:                " name
read -p "category of data  [train, test]:        " category
read -p "Nubr of CPU cores [8, 20]:              " NCPU

while true; do
	echo ""
	read -p "Change default intervals for phi, P and T?" yn
	echo ""
	case $yn in
		[Yy]* ) read -p "phi_0             [default: 0.50]     : " phi_0
		        read -p "phi_end           [default: 1.50]     : " phi_end
		        read -p "phi_step_size     [default: 0.50]     : " phi_step
                read -p "p_0               [default:   10]     : " p_0
                read -p "p_end             [default:   40]     : " p_end
                read -p "p_step_size       [default:   10]     : " p_step
		        read -p "start temperature [default:  650]     : " t_start
			    read -p "end temperature   [default: 1250]     : " t_end
			    read -p "temperature step  [default:   30]     : " t_step
			    python iterator_multi.py -mech $mechanism --pode $pode_n -nbr_run $name --category $category --phi_0 $phi_0 --phi_end $phi_end --phi_step $phi_step --p_0 $p_0 --p_end $p_end --p_step $p_step -t_0 $t_start -t_end $t_end -t_step $t_step --NCPU $NCPU; break;;
		[Nn]* ) python iterator_multi.py -mech $mechanism --pode $pode_n -nbr_run $name --category $category --NCPU $NCPU; exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done
