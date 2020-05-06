#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to plot iterator"
echo "------------------------"
echo ""

read -p "Choose plot         [ign_delay, thermo, species, HR, PV, time_scale]: " plot
echo ""

echo "Decide for which parameters the $plot plot should be performed"
read -p "PODE_n            [1, 2, 3, 4]:    " pode_n

case $plot in
		ign_delay ) read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                read -p "pressure          [10, 20, 40]:    " pressure
                while true; do
	               read -p "Change advance settings?          " yn
	               case $yn in
		               [Yy]* ) read -p "start temperature: " t_start
			                     read -p "end temperature:   " t_end
			                     read -p "temperature step:  " t_step
			                     python  plot_iterator.py -plt $plot -phi $equivalence_ratio -p $pressure --pode $pode_n -t_0 $t_start -t_end $t_end -t_step $t_step; break;;
		               [Nn]* ) python  plot_iterator.py -plt $plot -phi $equivalence_ratio -p $pressure --pode $pode_n ; exit;;
		               * ) echo "Please answer with yes or no." ;;
	                esac
                done; exit;;
		PV )         read -p "mechanism         [he, sun, cai]:  " mechanism
                 read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
                 while true; do
	                read -p "Change advance settings?          " yn
	                case $yn in
		                [Yy]* ) read -p "start temperature: " t_start
			                      read -p "end temperature:   " t_end
			                      read -p "temperature step:  " t_step
			                      read -p "temperature       ($t_start, $t_end, $t_step): " temperature
			                      python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -t_0 $t_start -t_end $t_end -t_step $t_step; break;;
		                [Nn]* ) read -p "temperature       (650, 1250, 15): " temperature
		                        python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature ; exit;;
		                 * ) echo "Please answer with yes or no." ;;
	                esac
                done; exit ;;
		time_scale ) read -p "mechanism         [he, sun, cai]:  " mechanism
                 read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
                 while true; do
	                read -p "Change advance settings?          " yn
	                case $yn in
		                [Yy]* ) read -p "start temperature: " t_start
			                      read -p "end temperature:   " t_end
			                      read -p "temperature step:  " t_step
			                      read -p "temperature       ($t_start, $t_end, $t_step): " temperature
			                      python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -t_0 $t_start -t_end $t_end -t_step $t_step; break;;
		                [Nn]* ) read -p "temperature       (650, 1250, 15): " temperature
		                        python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature ; exit;;
		                 * ) echo "Please answer with yes or no." ;;
	                esac
                done; exit ;;
		* )          read -p "mechanism         [he, sun, cai]:  " mechanism
                 read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
                 read -p "x axis scale      [PV, time]:      " scale
                 while true; do
	                read -p "Change advance settings?          " yn
	                case $yn in
		                [Yy]* ) read -p "start temperature: " t_start
			                      read -p "end temperature:   " t_end
			                      read -p "temperature step:  " t_step
			                      read -p "temperature       ($t_start, $t_end, $t_step): " temperature
			                      python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -x $scale -t_0 $t_start -t_end $t_end -t_step $t_step; break;;
		                [Nn]* ) read -p "temperature       (650, 1250, 15): " temperature
		                        python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio --pode $pode_n -p $pressure -t $temperature -x $scale; exit;;
		                 * ) echo "Please answer with yes or no." ;;
	                esac
                done; exit;;
esac
