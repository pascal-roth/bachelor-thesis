#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All plots in the BA of Pascal Roth will be created:"
echo "---------------------------------------------------"
echo ""

while true; do
	echo ""
	read -p "Does generate_data_train have been executed beforehand?" yn
	echo ""
	case $yn in
		[Yy]* ) echo "Very good"; break;;
		[Nn]* ) python ./00002-reactor-OME/iterator.py -mech cai -phi 0.5 1.0 1.5 -p 10 20 40 --pode 3 4-inf_print False;
		        exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done

echo "Additional necessary samples will be created"
echo ""
echo "Chapter 2: Chemical-Kinetic-Modelling"
echo "Data for IDTs-Exp-Comparison plot"
# Plot of the IDTs with experimental data for Phi=1.0, p = 10/20 bar and PODE 1-4

echo "Data for ..."


echo ""
echo ""
echo "Start with plot execution"
echo "-------------------------"
echo "Chapter 2: Chemical-Kinetic-Modelling"
echo "IDTs-Exp-Comparison plots"


