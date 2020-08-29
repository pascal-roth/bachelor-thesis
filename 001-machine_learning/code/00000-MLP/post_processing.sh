#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to MLP post processing"
echo "--------------------------"
echo ""

while true; do
    read -p "Processing method [loss, output, IDT, plt_train]:" method
    echo ""
    echo "Chose the trained model"
    read -p "nbr of network:                                  " name_net



    case $method in
        loss )     python post-processing.py --post $method -nbr_net $name_net; break;;
        output )   echo "Which test data should be used for the MLP"
                   read -p "mechanism         [he, sun, cai]:         " mechanism
                   read -p "name of test run  [XXX]:                  " name_test
                   while true; do
                       echo ""
                       read -p "Exclude data from testing (ar = array entry possible)?" yn
                       echo ""
                       case $yn in
                           [Yy]* ) read -p "PODE_n            [1, 2, 3, 4]     (ar):   " pode_n
                                   read -p "equivalence ratio (0.5, 1.5, 0.05) (ar):   " equivalence_ratio
                                   read -p "pressure          (10, 40, 2)      (ar):   " pressure
                                   read -p "temperature       (650, 1250, 30)  (ar):   " temp
                                   python post-processing.py --post $method -mech $mechanism -nbr_run $name_test -nbr_net $name_net  -phi $equivalence_ratio -p $pressure --pode $pode_n -temp $temp; break;;
                           [Nn]* ) python post-processing.py --post $method -mech $mechanism -nbr_run $name_test -nbr_net $name_net  ; exit;;
                           * ) echo "Please answer with yes or no." ;;
                       esac
                   done; break;;
        plt_train )echo "Select initial parameters"
                   read -p "PODE_n            [1, 2, 3, 4]     :       " pode_n
                   read -p "equivalence ratio (0.5, 1.5, 0.05) :       " equivalence_ratio
                   read -p "pressure          (10, 40, 2)      :       " pressure
                   read -p "temperature       (650, 1250, 30)  :       " temp
                   python post-processing.py --post $method -nbr_net $name_net -phi $equivalence_ratio -p $pressure --pode $pode_n -temp $temp; exit;;
        IDT )      echo "Which test data should be used for the MLP"
                   read -p "mechanism         [he, sun, cai]:         " mechanism
                   read -p "name of test run  [XXX]:                  " name_test
                   echo "Select initial parameters"
                   read -p "PODE_n            [1, 2, 3, 4]     :       " pode_n
                   read -p "equivalence ratio (0.5, 1.5, 0.05) :       " equivalence_ratio
                   read -p "pressure          (10, 40, 2)      :       " pressure
                   python post-processing.py --post $method -mech $mechanism -nbr_run $name_test -nbr_net $name_net  -phi $equivalence_ratio -p $pressure --pode $pode_n; exit;;
        * )        echo "chose a valid method"
    esac
done