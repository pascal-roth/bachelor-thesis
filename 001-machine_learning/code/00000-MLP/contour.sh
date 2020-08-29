#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to contour plot generator"
echo "---------------------------------"
echo ""

echo "Select parameters"
read -p "number of MLP                        [XXX]:                 " number_net
read -p "PODE_n                               [1, 2, 3, 4]:          " pode_n
read -p "equivalence ratio                    (0.5, 1.5, 0.05):      " phi
read -p "pressure                             ( 10,  20,    2):      " pressure

python contour.py --pode $pode_n -nbr_net $number_net -phi $phi -p $pressure;

exit