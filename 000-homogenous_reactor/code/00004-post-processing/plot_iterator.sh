echo "Welcome to plot iterator"
echo "------------------------"
echo ""

read -p "Choose plot         [ign_delays, thermo, species, HR, PV]: " plot
echo ""

echo "Decide for which parameters the $plot plot should be performed"
read -p "PODE_n            [1, 2, 3, 4]:    " pode_n

case $plot in
		ign_delays ) read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
			           python  plot_iterator.py -plt $plot -phi $equivalence_ratio -p $pressure --pode $pode_n; exit;;
		PV )         read -p "mechanism         [he, sun, cai]:  " mechanism
                 read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
                 read -p "temperature       (650, 1250, 15): " temperature
		             python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio -p $pressure -t_0 $temperature; exit;;
		* )          read -p "mechanism         [he, sun, cai]:  " mechanism
                 read -p "equivalence ratio [0.5, 1.0, 1.5]: " equivalence_ratio
                 read -p "pressure          [10, 20, 40]:    " pressure
                 read -p "temperature       (650, 1250, 15): " temperature
                 read -p "x axis scale      [PV, time]:      " scale
		             python plot_iterator.py -plt $plot -mech $mechanism -phi $equivalence_ratio -p $pressure -t_0 $temperature -x $scale; exit;;
esac
