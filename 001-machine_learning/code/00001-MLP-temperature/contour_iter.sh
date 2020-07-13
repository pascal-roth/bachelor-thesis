#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to contour plot iterator"
echo "--------------------------------"
echo ""

echo "Network 000:"
python contour.py --pode 3 -nbr_net 000 -phi 1.0 -p 20

echo "Network 001:"
python contour.py --pode 3 -nbr_net 001 -phi 1.0 -p 20

echo "Network 002:"
python contour.py --pode 3 -nbr_net 002 -phi 1.0 -p 20

echo "Network 003:"
python contour.py --pode 3 -nbr_net 003 -phi 1.0 -p 20

echo "Network 004:"
python contour.py --pode 3 -nbr_net 004 -phi 1.0 -p 20

echo "Network 005:"
python contour.py --pode 3 -nbr_net 005 -phi 1.0 -p 20

echo "Network 006:"
python contour.py --pode 3 -nbr_net 006 -phi 1.0 -p 20

echo "Network 007:"
python contour.py --pode 3 -nbr_net 007 -phi 1.0 -p 20