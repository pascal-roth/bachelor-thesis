#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All test data used in the BA of Pascal Roth will be created:"
echo "------------------------------------------------------------"
echo ""

echo "For Temperature MLP"
echo "Cai Mechanism with parameter setting that is different from the one found in the training samples"
# Interpolation of the equivalence ratio
python iterator.py -mech cai -phi 0.75 -p 20 --pode 3 -t_0  740 -t_end  741 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.95 -p 10 --pode 3 -t_0  950 -t_end  951 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.60 -p 40 --pode 3 -t_0  665 -t_end  666 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.25 -p 20 --pode 3 -t_0 1205 -t_end 1206 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.10 -p 10 --pode 3 -t_0  815 -t_end  816 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.37 -p 20 --pode 3 -t_0 1070 -t_end 1071 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.80 -p 40 --pode 3 -t_0  935 -t_end  936 -t_step 1 -nbr_run 001 --category test -inf_print

# Interpolation of the pressure
python iterator.py -mech cai -phi 1.0  -p 12 --pode 3 -t_0  680 -t_end  681 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.5  -p 17 --pode 3 -t_0  770 -t_end  771 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.0  -p 22 --pode 3 -t_0  980 -t_end  981 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.5  -p 25 --pode 3 -t_0 1115 -t_end 1116 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.5  -p 15 --pode 3 -t_0 1190 -t_end 1190 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.5  -p 30 --pode 3 -t_0  830 -t_end  831 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.0  -p 25 --pode 3 -t_0  920 -t_end  921 -t_step 1 -nbr_run 001 --category test -inf_print

# Interpolation of the temperature
python iterator.py -mech cai -phi 1.0  -p 10 --pode 3 -t_0 1000 -t_end 1001 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.5  -p 20 --pode 3 -t_0  685 -t_end  686 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.5  -p 20 --pode 3 -t_0  985 -t_end  986 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.0  -p 40 --pode 3 -t_0 1195 -t_end 1196 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.5  -p 10 --pode 3 -t_0 1080 -t_end 1080 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.0  -p 20 --pode 3 -t_0  840 -t_end  840 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.5  -p 40 --pode 3 -t_0  760 -t_end  760 -t_step 1 -nbr_run 001 --category test -inf_print

# Interpolation of all parameters
python iterator.py -mech cai -phi 1.25 -p 12 --pode 3 -t_0 1000 -t_end 1001 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.75 -p 17 --pode 3 -t_0  685 -t_end  686 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.95 -p 22 --pode 3 -t_0  985 -t_end  986 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.60 -p 25 --pode 3 -t_0 1195 -t_end 1196 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.10 -p 15 --pode 3 -t_0 1080 -t_end 1081 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.37 -p 30 --pode 3 -t_0  840 -t_end  841 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 0.80 -p 25 --pode 3 -t_0  760 -t_end  760 -t_step 1 -nbr_run 001 --category test -inf_print
python iterator.py -mech cai -phi 1.25 -p 19 --pode 3 -t_0  957 -t_end  958 -t_step 1 -nbr_run 001 --category test -inf_print
