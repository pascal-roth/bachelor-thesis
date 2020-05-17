#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to data iterator"
echo "------------------------"
echo ""

echo "Enter parameters of the homogeneous reactor (ar = array entry possible)"
read -p "mechanism         [he, sun, cai]:       " mechanism
read -p "PODE_n            [1, 2, 3, 4]    (ar): " pode_n
read -p "equivalence ratio [0.5, 1.0, 1.5] (ar): " equivalence_ratio
read -p "pressure          [10, 20, 40]    (ar): " pressure
read -p "name of run       [XXX]:                " name
read -p "category of data  [train, test]:        " category

while true; do
	echo ""
	read -p "Change temperature interval [default: (650, 1250, 15)]?" yn
	echo ""
	case $yn in
		[Yy]* ) read -p "start temperature: " t_start
			      read -p "end temperature:   " t_end
			      read -p "temperature step:  " t_step
			      python iterator.py -mech $mechanism -phi $equivalence_ratio -p $pressure --pode $pode_n -t_0 $t_start -t_end $t_end -t_step $t_step -nbr_run $name --category $category; break;;
		[Nn]* ) python iterator.py -mech $mechanism -phi $equivalence_ratio -p $pressure --pode $pode_n -nbr_run $name --category $category; exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done
