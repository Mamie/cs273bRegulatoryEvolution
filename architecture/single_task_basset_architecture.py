
import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress irrelevent warning messages
import keras;
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
from keras.constraints import maxnorm;
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import pandas
import time

start_time = time.time()

np.random.seed(1234) #sets random seed for consistency
seq_length = 2000 #2kb iput
num_labels = 1    #Our binary final prediction. The only label is the predicted probability that the gene is on or whatever
num_epochs = 2	  #Number of epochs to train for
save_folder = "Weights/"	#Folder where weights/parameters are stored
save_path = save_folder+"model_weights.h5" #Location to save weights/parameters to
##########################
init_weights_file = None#'Weights/model_weights.h5' #load in pre-trained weights/parameters Anna gave us here if we're using them. Elsewise, leave as None.
##########################
 
#Train Chromosomes 1 2 3 4 7 8 9 10 11 12 13 14 15 17 19 20 21 22
#Val Chromosomes 16 18
#Test Chromosomes  5 6
train = pandas.read_csv("train.csv", sep = ',').values 
val = pandas.read_csv("val.csv", sep = ',').values
test = pandas.read_csv("test.csv", sep = ',').values 

train = np.delete(train, 0, axis = 1)
val = np.delete(val, 0, axis =1)
test = np.delete(test, 0, axis =1)

np.random.shuffle(train)
np.random.shuffle(val)
np.random.shuffle(test)

print("Data read in after " + str(time.time()-start_time))


num_val_examples = len(val)
val_nums = np.empty([num_val_examples, 2001])
for i in range(num_val_examples):#len(train)):
	seq_list = list(val[i,0])
	for j in range(seq_length):
		if seq_list[j].lower() == 'a':
			val_nums[i, j] = 0
		elif seq_list[j].lower() == 'c':
			val_nums[i, j] = 1
		elif seq_list[j].lower() == 'g':
			val_nums[i, j] = 2
		elif seq_list[j].lower() == 't':
			val_nums[i, j] = 3
	val_nums[i, 2000] = val[i,1]
X_val_seq = val_nums[:, 0:2000]
Y_val = val_nums[:, 2000]

print("Val data converted after " + str(time.time()-start_time))


num_training_examples = 20000#len(train)/10

print(str(num_training_examples) + " training examples")

train_nums = np.empty([num_training_examples, 2001])
for i in range(num_training_examples):#len(train)):
	seq_list = list(train[i,0])
	for j in range(seq_length):
		if seq_list[j].lower() == 'a':
			train_nums[i, j] = 0
		elif seq_list[j].lower() == 'c':
			train_nums[i, j] = 1
		elif seq_list[j].lower() == 'g':
			train_nums[i, j] = 2
		elif seq_list[j].lower() == 't':
			train_nums[i, j] = 3
	train_nums[i, 2000] = train[i,1]
X_train_seq = train_nums[:, 0:2000]
Y_train = train_nums[:, 2000]

print("Train data converted after " + str(time.time()-start_time))



#Generate test data. This is just to test the architecture. X and Y will be loaded in from real data when this is actually used.
#X_train_seq = np.floor((np.random.rand(100,seq_length)*4)).astype(int)
#X_test_seq = np.floor((np.random.rand(100,seq_length)*4)).astype(int)

X_train = []
X_val = []
#Y_train = []
#Y_test = []

#Y_train = np.floor((np.random.rand(100)*2)).astype(int)
for i in range(num_training_examples):
	X_train.append(to_categorical(X_train_seq[i, :], num_classes=4))
X_train = np.moveaxis(X_train, 2, 0)
X_train = X_train.reshape(num_training_examples, 4, seq_length, 1)


#Y_val = np.floor((np.random.rand(100)*2)).astype(int)
for i in range(num_val_examples):
	X_val.append(to_categorical(X_val_seq[i, :], num_classes=4))
X_val = np.moveaxis(X_val, 2, 0)
X_val = X_val.reshape(num_val_examples, 4, seq_length, 1)

print("Converted to one-hot after " + str(time.time()-start_time))


#Build model based heavily on original Basset architecture
model = Sequential() #Multiple models can be made, each corresponding to a different species
model.add(Convolution2D(300,(4,19),input_shape=(4,int(seq_length), 1)))	
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Convolution2D(200,(1,11), kernel_constraint=maxnorm(7)))
model.add(BatchNormalization(axis=1))
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(1,4)))

model.add(Convolution2D(200,(1,7),kernel_constraint=maxnorm(7)))
model.add(BatchNormalization(axis=1))
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(1,4)))

model.add(Flatten())
model.add(Dense(1000,activity_regularizer=l1(0.00001),kernel_constraint=maxnorm(7)))
model.add(PReLU())
model.add(Dropout(0.3))

model.add(Dense(1000,activity_regularizer=l1(0.00001),kernel_constraint=maxnorm(7)))
model.add(PReLU())
model.add(Dropout(0.3))

model.add(Dense(num_labels)) 
model.add(Activation("sigmoid"))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="binary_crossentropy",optimizer=adam) 

#Load weights if they exist
if init_weights_file!=None:
	print("Restoring Weights")
	model.load_weights(init_weights_file)

#Train Model
model.fit(X_train, Y_train, batch_size=32, epochs=num_epochs, verbose=1) 

print("Trained after " + str(time.time()-start_time))

#Evaluate Model
y_probs = model.predict_proba(X_val)							#Predicted probability that each label is positive
AU_PRC = average_precision_score(Y_val, y_probs)				#Area under PR curve taken as average of precisions for all recalls
precision, recall, _ = precision_recall_curve(Y_val, y_probs)	#Arrays for plotting precision recall curve
print(AU_PRC)



#Plot Precision recall curve. Don't do this in Azure since there isn't really an interface for it to show plots there
'''
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(AU_PRC))
plt.savefig("AUPRC.png")
'''
#Save Parameters
if init_weights_file!=None:
    model.save_weights(init_weights_file)	
else:
	#Make new folder for weights if it doesn't yet exist
	try: 
	  os.makedirs(save_folder)
	except OSError as e:
	  if e.errno != errno.EEXIST:
	    raise 	
	model.save_weights(save_path)


#Display plots if they exist
print("Finished after " + str(time.time()-start_time))

#plt.show()


