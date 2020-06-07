#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All data used in the BA of Pascal Roth to create the exp-mech plots will be created:"
echo "------------------------------------------------------------------------------------"
echo ""

NCPU=4

echo "He Mechanism"
echo "------------"
python iterator_multi.py -mech he --pode 1 2 3 -nbr_run 000 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 20 --p_step 10 -t_0 650 -t_end 1250 -t_step 30 --NCPU $NCPU -inf_print
echo "Results for the cai exp data generated"

python iterator_multi.py -mech he --pode 3 -nbr_run 001 --category exp --phi_0 0.5 --phi_end 0.5 --phi_step 0.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.1111 --N2 0.8889 --NCPU $NCPU -inf_print
python iterator_multi.py -mech he --pode 3 -nbr_run 002 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0625 --N2 0.9375 --NCPU $NCPU -inf_print
python iterator_multi.py -mech he --pode 3 -nbr_run 003 --category exp --phi_0 1.5 --phi_end 1.5 --phi_step 1.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0476 --N2 0.9524 --NCPU $NCPU -inf_print
echo "Results for the he exp data generated"
echo ""


echo "Sun Mechanism"
echo "-------------"
python iterator_multi.py -mech sun --pode 1 2 3 -nbr_run 000 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 20 --p_step 10 -t_0 650 -t_end 1250 -t_step 30 --NCPU 4 -inf_print
echo "Results for the cai exp data generated"

python iterator_multi.py -mech sun --pode 3 -nbr_run 001 --category exp --phi_0 0.5 --phi_end 0.5 --phi_step 0.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.1111 --N2 0.8889 --NCPU $NCPU -inf_print
python iterator_multi.py -mech sun --pode 3 -nbr_run 002 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0625 --N2 0.9375 --NCPU $NCPU -inf_print
python iterator_multi.py -mech sun --pode 3 -nbr_run 003 --category exp --phi_0 1.5 --phi_end 1.5 --phi_step 1.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0476 --N2 0.9524 --NCPU $NCPU -inf_print
echo "Results for the he exp data generated"
echo ""

echo "Cai Mechanism"
echo "-------------"
python iterator_multi.py -mech cai --pode 1 2 3 4 -nbr_run 000 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 20 --p_step 10 -t_0 650 -t_end 1250 -t_step 30 --NCPU 4 -inf_print
echo "Results for the cai exp data generated"

python iterator_multi.py -mech cai --pode 3 -nbr_run 001 --category exp --phi_0 0.5 --phi_end 0.5 --phi_step 0.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.1111 --N2 0.8889 --NCPU $NCPU -inf_print
python iterator_multi.py -mech cai --pode 3 -nbr_run 002 --category exp --phi_0 1.0 --phi_end 1.0 --phi_step 1.0 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0625 --N2 0.9375 --NCPU $NCPU -inf_print
python iterator_multi.py -mech cai --pode 3 -nbr_run 003 --category exp --phi_0 1.5 --phi_end 1.5 --phi_step 1.5 --p_0 10 --p_end 15 --p_step 5 -t_0 650 -t_end 1250 -t_step 30 --O2 0.0476 --N2 0.9524 --NCPU $NCPU -inf_print
echo "Results for the he exp data generated"
echo ""


