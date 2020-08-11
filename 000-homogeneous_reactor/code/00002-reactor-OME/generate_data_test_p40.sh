#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All test data used in the BA of Pascal Roth will be created:"
echo "------------------------------------------------------------"
echo ""

echo "For Temperature MLP"
echo "Cai Mechanism with parameter setting that is different from the one found in the training samples"

# Test data to plot (include one interpolation for every parameter and one interpolation for all parameters)
# especially for set 3 where the pressure is pre-selected
#python iterator.py -mech cai --pode 3 -nbr_run 003 --category test --phi_0 0.77 --phi_end 0.77 --phi_step 0.77 --p_0 40 --p_end 40 --p_step 40 -t_0  740 -t_end  740 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 003 --category test --phi_0 1.00 --phi_end 1.00 --phi_step 1.00 --p_0 40 --p_end 40 --p_step 40 -t_0  955 -t_end  955 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 003 --category test --phi_0 0.67 --phi_end 0.67 --phi_step 0.67 --p_0 40 --p_end 40 --p_step 40 -t_0 1065 -t_end 1065 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 003 --category test --phi_0 1.27 --phi_end 1.27 --phi_step 1.27 --p_0 40 --p_end 40 --p_step 40 -t_0 1175 -t_end 1175 -t_step 30

# Test data to compare the IDTs
python iterator_multi.py -mech cai --pode 3 -nbr_run 004 --category test --phi_0 0.67 --phi_end 0.67 --phi_step 0.67 --p_0 40 --p_end 40 --p_step 40 -t_0 665 -t_end 1235 -t_step 30

