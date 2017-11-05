#!/usr/bin/python
# -*- coding: <encoding name> -*-
"""
    File name: convertPeakcall2BED3.py
    Author: Mamie Wang
    Date created: 11/05/2017
    Date last modified: 11/05/2017
    Python version: 3.6

    Input: Peak call file
    Output: BED3 file (first three columns of the peak call file)
"""


if __name__=='__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filePath')
    argparser.add_argument('outFilePath')
    arguments = argparser.parse_args()
    filePath = arguments.filePath
    outFilePath = arguments.outFilePath
    with open(filePath, 'r') as infile:
        content = infile.readlines()
    content = list(content)
    content = [line.rstrip().split('\t')[:3] for line in content]
    with open(outFilePath, 'w') as outfile:
        for line in content:
            outfile.write('{0}\t{1}\t{2}\n'.format(*line))
