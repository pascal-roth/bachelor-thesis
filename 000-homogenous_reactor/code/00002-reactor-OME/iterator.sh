echo "Enter parameters of the homogeneous reactor"
read -p "mechanism         [he, sun, cai]:           " mechanism
read -p "PODE_n            [1, 2, 3, 4]:             " pode_n
read -p "equivalence ratio [0.5, 1.0, 1.5, all=0.0]: " equivalence_ratio
read -p "pressure          [10, 20, 40, all=0]:      " pressure

while true; do
	read -p "Change temperature interval [default: (650, 1250, 15)]?" yn
	case $yn in
		[Yy]* ) read -p "start temperature: " t_start
			read -p "end temperature: " t_end
			read -p "temperature step:" t_step
			python -m iterator.py -mech $mechansim -phi $equivalence_ratio -p $pressure --pode $pode_n -t_0 $t_start -t_end $t_end -t_step $t_step; break;;
		[Nn]* ) python -m iterator.py -mech $mechanism -phi $equivalence_ratio -p $pressure --pode $pode_n ; exit;;
		* ) echo "Please answer with yes or no." ;;
	esac
done
