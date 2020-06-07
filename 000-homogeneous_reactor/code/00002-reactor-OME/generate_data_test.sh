#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All test data used in the BA of Pascal Roth will be created:"
echo "------------------------------------------------------------"
echo ""

echo "For Temperature MLP"
echo "Cai Mechanism with parameter setting that is different from the one found in the training samples"
# Test data to plot (include one interpolation for every parameter and one interpolation for all parameters)
python iterator.py -mech cai --pode 3 -nbr_run 000 --category test --phi_0 0.77 --phi_end 0.77 --phi_step 0.77 --p_0 10 --p_end 10 --p_step 10 -t_0  740 -t_end  740 -t_step 30
python iterator.py -mech cai --pode 3 -nbr_run 000 --category test --phi_0 1.00 --phi_end 1.00 --phi_step 1.00 --p_0 17 --p_end 17 --p_step 17 -t_0  950 -t_end  950 -t_step 30
python iterator.py -mech cai --pode 3 -nbr_run 000 --category test --phi_0 1.50 --phi_end 1.50 --phi_step 1.50 --p_0 20 --p_end 20 --p_step 20 -t_0 1065 -t_end 1065 -t_step 30
python iterator.py -mech cai --pode 3 -nbr_run 000 --category test --phi_0 1.27 --phi_end 1.27 --phi_step 1.27 --p_0 25 --p_end 25 --p_step 25 -t_0 1175 -t_end 1175 -t_step 30

## Interpolation of the equivalence ratio
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.77 --phi_end 0.77 --phi_step 0.77 --p_0 16 --p_end 16 --p_step 2 -t_0  740 -t_end  740 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.93 --phi_end 0.93 --phi_step 0.93 --p_0 28 --p_end 28 --p_step 2 -t_0  950 -t_end  950 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.62 --phi_end 0.62 --phi_step 0.62 --p_0 12 --p_end 12 --p_step 2 -t_0  680 -t_end  680 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.27 --phi_end 1.27 --phi_step 1.27 --p_0 22 --p_end 22 --p_step 2 -t_0 1190 -t_end 1190 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.08 --phi_end 1.08 --phi_step 1.08 --p_0 32 --p_end 32 --p_step 2 -t_0  830 -t_end  830 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.37 --phi_end 1.37 --phi_step 1.37 --p_0 26 --p_end 26 --p_step 2 -t_0 1070 -t_end 1070 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.82 --phi_end 0.82 --phi_step 0.82 --p_0 38 --p_end 38 --p_step 2 -t_0  920 -t_end  920 -t_step 30
#
## Interpolation of the pressure
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.70 --phi_end 0.70 --phi_step 0.70 --p_0 17 --p_end 17 --p_step 2 -t_0  860 -t_end  860 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.90 --phi_end 0.90 --phi_step 0.90 --p_0 29 --p_end 29 --p_step 2 -t_0  950 -t_end  950 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.60 --phi_end 0.60 --phi_step 0.60 --p_0 13 --p_end 13 --p_step 2 -t_0  680 -t_end  680 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.20 --phi_end 1.20 --phi_step 1.20 --p_0 23 --p_end 23 --p_step 2 -t_0 1190 -t_end 1190 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.00 --phi_end 1.00 --phi_step 1.00 --p_0 31 --p_end 31 --p_step 2 -t_0  830 -t_end  830 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.40 --phi_end 1.40 --phi_step 1.40 --p_0 27 --p_end 27 --p_step 2 -t_0 1070 -t_end 1070 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.80 --phi_end 0.80 --phi_step 0.80 --p_0 37 --p_end 37 --p_step 2 -t_0  920 -t_end  920 -t_step 30
#
## Interpolation of the temperature
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.70 --phi_end 0.70 --phi_step 0.70 --p_0 16 --p_end 16 --p_step 2 -t_0  845 -t_end  845 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.90 --phi_end 0.90 --phi_step 0.90 --p_0 28 --p_end 28 --p_step 2 -t_0  965 -t_end  965 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.60 --phi_end 0.60 --phi_step 0.60 --p_0 12 --p_end 12 --p_step 2 -t_0  695 -t_end  695 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.20 --phi_end 1.20 --phi_step 1.20 --p_0 24 --p_end 24 --p_step 2 -t_0 1175 -t_end 1175 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.00 --phi_end 1.00 --phi_step 1.00 --p_0 32 --p_end 32 --p_step 2 -t_0  815 -t_end  815 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.40 --phi_end 1.40 --phi_step 1.40 --p_0 20 --p_end 20 --p_step 2 -t_0 1065 -t_end 1065 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.80 --phi_end 0.80 --phi_step 0.80 --p_0 38 --p_end 38 --p_step 2 -t_0  725 -t_end  725 -t_step 30
#
## Interpolation of all parameters
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.77 --phi_end 0.77 --phi_step 0.77 --p_0 17 --p_end 17 --p_step 2 -t_0  845 -t_end  845 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.93 --phi_end 0.93 --phi_step 0.93 --p_0 29 --p_end 29 --p_step 2 -t_0  965 -t_end  965 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.62 --phi_end 0.62 --phi_step 0.62 --p_0 13 --p_end 13 --p_step 2 -t_0  695 -t_end  695 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.27 --phi_end 1.27 --phi_step 1.27 --p_0 25 --p_end 25 --p_step 2 -t_0 1175 -t_end 1175 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.08 --phi_end 1.08 --phi_step 0.08 --p_0 33 --p_end 33 --p_step 2 -t_0  815 -t_end  815 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 1.37 --phi_end 1.37 --phi_step 1.37 --p_0 21 --p_end 21 --p_step 2 -t_0 1065 -t_end 1065 -t_step 30
#python iterator.py -mech cai --pode 3 -nbr_run 001 --category test --phi_0 0.82 --phi_end 0.82 --phi_step 0.82 --p_0 37 --p_end 37 --p_step 2 -t_0  725 -t_end  725 -t_step 30
