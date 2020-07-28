#!/bin/bash

source /home/pascal/anaconda3/bin/activate BA

echo "Welcome to contour plot iterator"
echo "--------------------------------"
echo ""

pode=3
equivalence_ratio=1.0
pressure=20
diff=both

#echo "Network 000:"
#python contour.py --pode $pode -nbr_net 000 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 001:"
#python contour.py --pode $pode -nbr_net 001 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 002:"
#python contour.py --pode $pode -nbr_net 002 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 003:"
#python contour.py --pode $pode -nbr_net 003 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 004:"
#python contour.py --pode $pode -nbr_net 004 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 005:"
#python contour.py --pode $pode -nbr_net 005 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 006:"
#python contour.py --pode $pode -nbr_net 006 -phi $equivalence_ratio -p $pressure -diff $diff
#
#echo "Network 007:"
#python contour.py --pode $pode -nbr_net 007 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 008:"
python contour.py --pode $pode -nbr_net 008 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 009:"
python contour.py --pode $pode -nbr_net 009 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 010:"
python contour.py --pode $pode -nbr_net 010 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 011:"
python contour.py --pode $pode -nbr_net 011 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 012:"
python contour.py --pode $pode -nbr_net 012 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 013:"
python contour.py --pode $pode -nbr_net 013 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 014:"
python contour.py --pode $pode -nbr_net 014 -phi $equivalence_ratio -p $pressure -diff $diff

echo "Network 015:"
python contour.py --pode $pode -nbr_net 015 -phi $equivalence_ratio -p $pressure -diff $diff