#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "All data used in the BA of Pascal Roth to create the exp-mech plots will be created:"
echo "------------------------------------------------------------------------------------"
echo ""

echo "He Mechanism"
echo "------------"
python iterator.py -mech he -phi 1.0 -p 10 20 --pode 1 2 3 -nbr_run 000 --category exp -inf_print
echo "Results for the cai exp data generated"

python iterator.py -mech he -phi 0.5 -p 10 15 --pode 3 -nbr_run 001 --category exp --O2 0.1111 --N2 0.8889 -inf_print
python iterator.py -mech he -phi 1.0 -p 10 15 --pode 3 -nbr_run 002 --category exp --O2 0.0625 --N2 0.9375 -inf_print
python iterator.py -mech he -phi 1.5 -p 10 15 --pode 3 -nbr_run 003 --category exp --O2 0.0476 --N2 0.9524 -inf_print
echo "Results for the he exp data generated"
echo ""


echo "Sun Mechanism"
echo "-------------"
python iterator.py -mech sun -phi 1.0 -p 10 20 --pode 1 2 3 -nbr_run 000 --category exp -inf_print
echo "Results for the cai exp data generated"

python iterator.py -mech sun -phi 0.5 -p 10 15 --pode 3 -nbr_run 001 --category exp --O2 0.1111 --N2 0.8889 -inf_print
python iterator.py -mech sun -phi 1.0 -p 10 15 --pode 3 -nbr_run 002 --category exp --O2 0.0625 --N2 0.9375 -inf_print
python iterator.py -mech sun -phi 1.5 -p 10 15 --pode 3 -nbr_run 003 --category exp --O2 0.0476 --N2 0.9524 -inf_print
echo "Results for the he exp data generated"
echo ""

echo "Cai Mechanism"
echo "-------------"
python iterator.py -mech cai -phi 1.0 -p 10 20 --pode 1 2 3 4 -nbr_run 000 --category exp -inf_print
echo "Results for the cai exp data generated"

python iterator.py -mech cai -phi 0.5 -p 10 15 --pode 3 -nbr_run 001 --category exp --O2 0.1111 --N2 0.8889 -inf_print
python iterator.py -mech cai -phi 1.0 -p 10 15 --pode 3 -nbr_run 002 --category exp --O2 0.0625 --N2 0.9375 -inf_print
python iterator.py -mech cai -phi 1.5 -p 10 15 --pode 3 -nbr_run 003 --category exp --O2 0.0476 --N2 0.9524 -inf_print
echo "Results for the he exp data generated"
echo ""


