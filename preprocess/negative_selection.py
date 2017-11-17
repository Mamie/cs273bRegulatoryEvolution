import pandas as pd
import numpy as np
import glob
import csv
import re
import sys
#run script by like so: python negative_selectoin.py "path_of_chipseq_files"-> it will output the files to the same path
#where path_of_chipseq_files is the folder containing all (and only) the chipseq files

#create a list object with all of the file names to be preprocessed (all chip-seq data sets)
#need to give it the path with the folder containing only the chipseq files
chip_file_path = sys.argv[1]
files=glob.glob(chip_file_path+"/*")
subs=files[0]
subs2=re.sub('/[^/]*$','',subs)

#function to create the negative set for a given file
def neg_set(match):
    enhancer = subs2+'/'+match[0]
    promoter = subs2+'/'+match[1]
    print enhancer
    e0 = pd.read_csv(enhancer, sep='\t')
    p0 = z0=pd.read_csv(promoter, sep='\t')
    #take the union of the enhancer and promoter data sets (the complement will be the negative set)
    z0 = pd.concat([e0,p0])
    #Subtract 1kb from start and add 1kb to end of each promoter/enhancer
    z0['Start'] = z0['Start'] - 1000
    z0['End'] = z0['End'] + 1000
    z0=z0[z0['Start']>0]

    chroms=list(z0['Chrom'].unique())

    all_negs_set=[]

    if len(chroms)<40:
        for j in chroms:
            z1=z0[z0['Chrom']==j]
            pairs=zip(z1['Chrom'].values,z1['Start'].values, z1['End'].values)
            pairs.sort(key=lambda interval: interval[1])
            listpairs = [list(elem) for elem in pairs]
            merged = [listpairs[0]]
            for current in listpairs:
                previous = merged[-1]
                if current[1] <= previous[2]:
                    previous[2] = max(previous[2], current[2])
                else:
                    merged.append(current)
            neg_set = [[merged[i][0],merged[i][2],merged[i+1][1]] for i in range(0,len(merged)-1)]
            all_negs_set.append(neg_set)

        flat_neg_set = [item for sublist in all_negs_set for item in sublist]
        #create negative set of 2kb regions at beginning and end of the negative complement regions, combine them together
        kb2_set_1 = [[i[0],i[1],i[1]+2000] for i in flat_neg_set if i[1]+2000 < i[2]]
        kb2_set_2 = [[i[0],i[2]-2000,i[2]] for i in flat_neg_set if i[2]-2000 > i[1]]
        kb_2_full = kb2_set_1 + kb2_set_2
        kb_2_full.sort()
        print len(kb_2_full)

        with open(subs2+"/"+match[0]+"_neg_set.bed","wb") as f:
            writer = csv.writer(f,delimiter='\t')
            writer.writerows(kb_2_full)
    else:
        z1=z0
        pairs=zip(z1['Chrom'].values,z1['Start'].values, z1['End'].values)
        pairs.sort(key=lambda interval: interval[1])
        listpairs = [list(elem) for elem in pairs]
        merged = [listpairs[0]]
        for current in listpairs:
            previous = merged[-1]
            if current[1] <= previous[2]:
                previous[2] = max(previous[2], current[2])
            else:
                merged.append(current)

        neg_set = [[merged[i][0],merged[i][2],merged[i+1][1]] for i in range(0,len(merged)-1)]
        all_negs_set.append(neg_set)

        flat_neg_set = [item for sublist in all_negs_set for item in sublist]
        kb2_set_1 = [[i[0],i[1],i[1]+2000] for i in flat_neg_set if i[1]+2000 < i[2]]
        kb2_set_2 = [[i[0],i[2]-2000,i[2]] for i in flat_neg_set if i[2]-2000 > i[1]]
        kb_2_full = kb2_set_1 + kb2_set_2
        kb_2_full.sort()
        print len(kb_2_full)

        with open(subs2+"/"+match[0]+"_neg_set.bed","wb") as f:
            writer = csv.writer(f,delimiter='\t')
            writer.writerows(kb_2_full)


#data preprocessing-> match the enhancer and promoter files that go together
names=[re.split('/', i)[-1] for i in files]
matches=[]
for  i in names:
    for j in names:
        if re.split('-', i)[0]==re.split('-', j)[0] and i!=j:
            sort_pair = [i,j]
            sort_pair.sort()
            matches.append(sort_pair)
tuple_of_tuples = tuple(tuple(x) for x in matches)
set_of_matches = set(tuple_of_tuples)
matches_unique_sorted = [list(i) for i in set_of_matches]

#for loop to apply the function neg_set over all of the chipseq files
for j in matches_unique_sorted:
    print j
    neg_set(j)

