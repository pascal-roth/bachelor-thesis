#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All training data used in the BA of Pascal Roth will be created:"
echo "----------------------------------------------------------------"
echo ""

#echo "Cai Mechanism with parameter setting: PODE =[3], Phi = (0.5, 1.5, 20), P = (10, 20, 40)bar for a temperature range (650, 1250, 30)"
#python iterator.py -mech cai --pode 3 -nbr_run 000 --category train --phi_0 0.5 --phi_end 1.5 --phi_step 0.05 --p_0 10 --p_end 40 --p_step 2 -t_0 650 -t_end 1250 -t_step 30

echo "Cai Mechanism with parameter setting: PODE = [3, 4] Phi = (0.5, 1.5, 20), P = 40bar for a temperature range (650, 1250,15)"
python iterator.py -mech cai --pode 3 4 -nbr_run 001 --category train --phi_0 0.5 --phi_end 1.5 --phi_step 0.05 --p_0 40 --p_end 40 --p_step 40 -t_0 650 -t_end 1250 -t_step 30
