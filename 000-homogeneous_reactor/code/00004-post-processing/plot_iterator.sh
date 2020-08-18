#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to plot iterator"
echo "------------------------"
echo ""

read -p "Choose plot         [ign_delay, thermo, species, HRR, PV, time_scale, T+HRR]: " plot
echo ""

echo "Decide which data should be used for the $plot plot"
read -p "category of data  [train, test]:      " category
read -p "name of run       [XXX]:              " name
echo ""

echo "Decide for which parameters the $plot plot should be performed"
read -p "PODE_n            [1, 2, 3, 4]:       " pode_n

case $plot in
		ign_delay )  read -p "mechanism         [he, sun, cai, all]:" mechanism
                 read -p "equivalence ratio (0.5, 1.5, 0.05):   " equivalence_ratio
                 read -p "pressure          (10, 40, 10):       " pressure
                 python  plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio -p $pressure --pode $pode_n -nbr_run $name --category $category; exit;;
		PV )         read -p "mechanism         [he, sun, cai]:     " mechanism
                 read -p "equivalence ratio (0.5, 1.5, 0.05):   " equivalence_ratio
                 read -p "pressure          (10, 40, 10):       " pressure
                 read -p "temperature       (650, 1250, 30):    " temperature
		             python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -nbr_run $name --category $category; exit;;
		time_scale ) read -p "mechanism         [he, sun, cai]:     " mechanism
                 read -p "equivalence ratio (0.5, 1.5, 0.05):   " equivalence_ratio
                 read -p "pressure          (10, 40, 10):       " pressure
                 read -p "temperature       (650, 1250, 30):    " temperature
		             python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -nbr_run $name --category $category; exit;;
		* )          read -p "mechanism         [he, sun, cai]:     " mechanism
                 read -p "x axis scale      [PV, time]:         " scale
                 read -p "equivalence ratio (0.5, 1.5, 0.05):   " equivalence_ratio
                 read -p "pressure          (10, 40, 10):       " pressure
                 read -p "temperature       (650, 1250, 30):    " temperature
		             python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -x $scale -nbr_run $name --category $category; exit;;
esac
