
#These are where you put the file names for the positive and negative fasta files
#EDIT HERE FOR EACH SPECIES
##############################################################################
inputFilenameP = "hsa-enhancer_replicated_peaks_bedtsv.fasta" #Positive
inputFilenameN = "hsa-H3K27Ac_replicated-peaks_macs_neg_set.bed.fasta" #negative
##############################################################################

#Create positive chromosome files
with open(inputFilenameP) as f:
	for line in f:
		if (any(char.isdigit() for char in line[:10])):
			i = line.index(":")
			outputFile = open(str(line[1:i]+"P.csv") , 'a')
			outputFile.write(line[:i] + ",")
		else:
			outputFile.write(line)
			outputFile.close()

#Create negative chromosome files
with open(inputFilenameN) as f:
	for line in f:
		if (any(char.isdigit() for char in line[:10])):
			i = line.index(":")
			outputFile = open(str(line[1:i]+"N.csv") , 'a')
			outputFile.write(line[:i] + ",")
		else:
			outputFile.write(line)
			outputFile.close()


#Each species has different chromosomes so for each one, there should be two left out for validation adn two left out for testing. 
#Which chromosomes are in each dataset should be selected at random and will change from species to species
#EDIT HERE FOR EACH SPECIES
##############################################################################
train = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22]
val = [16, 18]
test = [5, 6]
##############################################################################


#Positive data set
train_output = "positive_train.csv"
val_output = "positive_val.csv"
test_output = "positive_test.csv"

outputFile = open(train_output, 'a')
for i in train:
	inputFile = open("chr"+str(i)+"P.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",1\n")
	inputFile.close()
outputFile.close()

outputFile = open(val_output, 'a')
for i in val:
	inputFile = open("chr"+str(i)+"P.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",1\n")
	inputFile.close()
outputFile.close()

outputFile = open(test_output, 'a')
for i in test:
	inputFile = open("chr"+str(i)+"P.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",1\n")
	inputFile.close()
outputFile.close()

#Negtive data set
train_output = "negative_train.csv"
val_output = "negative_val.csv"
test_output = "negative_test.csv"

outputFile = open(train_output, 'a')
for i in train:
	inputFile = open("chr"+str(i)+"N.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",0\n")
	inputFile.close()
outputFile.close()

outputFile = open(val_output, 'a')
for i in val:
	inputFile = open("chr"+str(i)+"N.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",0\n")
	inputFile.close()
outputFile.close()

outputFile = open(test_output, 'a')
for i in test:
	inputFile = open("chr"+str(i)+"N.csv")
	for line in inputFile:
		outputFile.write(line.rstrip("\n") +",0\n")
	inputFile.close()
outputFile.close()

#Combine positive and negative sets
train_output = "train.csv"
val_output = "val.csv"
test_output = "test.csv"

outputFile = open(train_output, 'a')
inputFile = open("positive_train.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
inputFile = open("negative_train.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
outputFile.close()

outputFile = open(val_output, 'a')
inputFile = open("positive_val.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
inputFile = open("negative_val.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
outputFile.close()

outputFile = open(test_output, 'a')
inputFile = open("positive_test.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
inputFile = open("negative_test.csv")
for line in inputFile:
	outputFile.write(line)
inputFile.close()
outputFile.close()
