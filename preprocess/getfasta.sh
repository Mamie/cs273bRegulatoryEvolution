#!/bin/bash
# -*- coding: <encoding name> -*-
# File name: getfasta.sh 
# Author: Mamie Wang
# Date created: 11/09/2017
# Date last modified: 11/13/2017

# Input:
#	a: A text file containing path to the reference fasta file
#	b: A text file containing path to corresponding bed3 interval file
# Output:
#	sequences corresponding to the intervals in a

 
while read -r a && read -r b <&3; do
 >&2 echo "Task: $a"
 bedtools getfasta -fi $a -bed $b -fo "$b.fasta";
done < $1 3<$2



