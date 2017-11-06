import pandas as pd
import glob
import csv

#run script like so: python negative_selectoin.py "path_of_chipseq_files"
#where path_of_chipseq_files is the folder containing all (and only) the chipseq files
chip_file_path = sys.argv[1]

files=glob.glob(chip_file_path+"*")

#function to create the negative set for a given file
def neg_set(file_name):
    z0=pd.read_csv(file_name, sep='\t')
    chroms=list(z0['Chrom'].unique())
    all_negs_set=[]
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

    #write the negative set to a file with the "neg_set.bed" extension on top of the existing file name in the same path
    with open(file_name+"neg_set.bed", "wb") as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(flat_neg_set)

#for loop to apply the function neg_set over all of the chipseq files
for j in files:
    neg_set(j)
