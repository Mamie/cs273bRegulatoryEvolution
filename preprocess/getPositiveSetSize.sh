#!/bin/bash
# -*- coding: <encoding name> -*-
# File name: getPositiveSetSize.sh 
# Author: Mamie Wang
# Date created: 11/15/2017
# Date last modified: 11/15/2017

# Input:
#	path to the fasta files
# Output:
#	the number of sequences in the fasta file


wc -l `find $1 -name "*.fasta"` 


