#!/usr/bin/python
# -*- coding: <encoding name> -*-
"""
    File name: createSlidingWindow.py
    Author: Mamie Wang
    Date created: 11/02/2017
    Date last modified: 11/02/2017
    Python version: 3.6

    Input: BED3 file of the summit region of the peak call file
    Output: BED3 file of all sliding window intervals with size 2 kb, step size 500bp that have more than half of it inside each summit region, written as 'positiveSet.tsv'
"""

import math

def createSlidingWindow(interval, s=2000, k=500):
    Y = []
    start = interval[0]
    end = interval[1]
    a = max(1, int(start - math.floor(s/2)))
    b = a + s
    while a <= end - s/2:
        overlapDistance = min(b, end) - max(a, start) + 1
        if overlapDistance >= s/2:
            Y.append([a, b])
        a += k
        b += k
    return Y

def parseBED3(filePath):
    with open(filePath, 'r') as infile:
        bed3 = infile.readlines()
    bed3 = list(bed3)[1:]
    bed3 = [list(map(int, line.rstrip().split('\t'))) for line in bed3]
    return bed3

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filePath')
    args = parser.parse_args()
    bed3 = parseBED3(args.filePath)
    Y = []
    for i in range(len(bed3)):
        y = createSlidingWindow(bed3[i][1:3])
        y = [[bed3[i][0]] + interval for interval in y]
        Y = Y + y
    with open('positiveSet.tsv', 'w') as outfile:
        outfile.write('Chrom   start   end\n')
        for y in Y:
            outfile.write(("{0}\t{1}\t{2}\n").format(*y))

