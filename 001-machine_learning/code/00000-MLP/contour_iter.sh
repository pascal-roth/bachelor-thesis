#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to contour plot iterator"
echo "--------------------------------"
echo ""

pode=3
equivalence_ratio=1.0
pressure=40

echo "Network 000:"
python contour.py --pode $pode -nbr_net 000 -phi $equivalence_ratio -p $pressure

echo "Network 001:"
python contour.py --pode $pode -nbr_net 001 -phi $equivalence_ratio -p $pressure

echo "Network 002:"
python contour.py --pode $pode -nbr_net 002 -phi $equivalence_ratio -p $pressure

echo "Network 003:"
python contour.py --pode $pode -nbr_net 003 -phi $equivalence_ratio -p $pressure

echo "Network 004:"
python contour.py --pode $pode -nbr_net 004 -phi $equivalence_ratio -p $pressure

echo "Network 005:"
python contour.py --pode $pode -nbr_net 005 -phi $equivalence_ratio -p $pressure

echo "Network 006:"
python contour.py --pode $pode -nbr_net 006 -phi $equivalence_ratio -p $pressure

echo "Network 007:"
python contour.py --pode $pode -nbr_net 007 -phi $equivalence_ratio -p $pressure
