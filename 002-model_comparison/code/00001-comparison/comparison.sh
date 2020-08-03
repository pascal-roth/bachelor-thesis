#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to the comparison script between the MLP and the GRM (Global Reaction Mechanism)"
echo "----------------------------------------------------------------------------------------"
echo ""
echo "Import Notice!: Before starting the script the HR has to be executed with the Phi, Pressure and Temperature values which should be compared. The results should be saved in a test file (if the test run nbr is different from 002 the python file has to be adjusted). Furthermore the network networks number for the different cases have to be adjusted in the python script"
echo ""

while true; do
    echo "Select intended comparison option:"
    read -p "comparison:        ['IDT', 'thermo', 'educts', 'products']:" comparison

    case $comparison in
        IDT ) read -p "mechanism          [he, sun, cai]   :                      " mechanism
              read -p "PODE_n             [1, 2, 3, 4]     :                      " pode_n
              read -p "equivalence ratio  [0.5, 0.67, 1.0] :                      " equivalence_ratio
              read -p "pressure           (10, 40, 2)      :                      " pressure
              python comparison.py --comparison $comparison -mech $mechanism --pode $pode_n -phi $equivalence_ratio -p $pressure; break;;
        * )   read -p "mechanism          [he, sun, cai]   :                      " mechanism
              read -p "PODE_n             [1, 2, 3, 4]     :                      " pode_n
              read -p "equivalence ratio  [0.5, 0.67, 1.0] :                      " equivalence_ratio
              read -p "pressure           (10, 40, 2)      :                      " pressure
              read -p "temperature        (680, 1240, 40)  :                      " temp
              python comparison.py --comparison $comparison -mech $mechanism --pode $pode_n -phi $equivalence_ratio -p $pressure -T $temp; exit;;
    esac
done
