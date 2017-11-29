
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
import pydot
import graphviz
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#import tensorflow as tf




start_time = time.time()

np.random.seed(1234) #sets random seed for consistency
seq_length = 2000 #2kb iput
num_labels = 1    #Our binary final prediction. The only label is the predicted probability that the gene is on or whatever
num_epochs = 10	  #Number of epochs to train for
save_folder = "Weights/"	#Folder where weights/parameters are stored
save_path = save_folder+"model_weights.h5" #Location to save weights/parameters to
##########################
init_weights_file = None#'Weights/model_weights.h5' #load in pre-trained weights/parameters Anna gave us here if we're using them. Elsewise, leave as None.
##########################
 
#Train Chromosomes 1 2 3 4 7 8 9 10 11 12 13 14 15 17 19 20 21 22
#Val Chromosomes 16 18
#Test Chromosomes  5 6

train = pandas.read_csv("hsa_train.csv", sep = ',').values 
print("Training data read in after"+ str(time.time()-start_time))
val = pandas.read_csv("hsa_val.csv", sep = ',').values
#test = pandas.read_csv("test.csv", sep = ',').values 

train = np.delete(train, 0, axis = 1)
val = np.delete(val, 0, axis =1)
#test = np.delete(test, 0, axis =1)

np.random.shuffle(train)
np.random.shuffle(val)
#np.random.shuffle(test)

print("All data read in after " + str(time.time()-start_time))


num_val_examples = min(20000,len(val))
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


num_training_examples = min(20000,len(train))

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


X_train = []
X_val = []

for i in range(num_training_examples):
	X_train.append(to_categorical(X_train_seq[i, :], num_classes=4))

#print(np.shape(X_train))
X_train = np.asarray(X_train)
#X_train = np.moveaxis(X_train, 2, 0) #These reshapes might be causing problemsX_train = tf.convert_to_tensor(X_train)
#print(np.shape(X_train))


X_train = np.expand_dims(X_train, axis = 1) #These reshapes might be causing problems should be (num_examples, 1, sequence_length, 4)

print(np.shape(X_train))


for i in range(num_val_examples):
	X_val.append(to_categorical(X_val_seq[i, :], num_classes=4))
X_val = np.asarray(X_val)
X_val = np.expand_dims(X_val, axis = 1) #These reshapes might be causing problems should be (num_examples, 1, sequence_length, 4)

print("Converted to one-hot after " + str(time.time()-start_time))


#Build model based heavily on original Basset architecture
model = Sequential() #Multiple models can be made, each corresponding to a different species
model.add(Convolution2D(300,(1,19),input_shape=(1,int(seq_length), 4))) #the 1 and the 4 should be switched here. Correct order would be 300,(1,19) and (1,int(seq_length), 4)))
model.add(BatchNormalization(axis=-1))	#Isn't in all versions but is good
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Convolution2D(filters=200,kernel_size=(1,11)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,4)))

model.add(Convolution2D(filters=200,kernel_size=(1,7)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,4)))

model.add(Flatten())
model.add(Dense(1000))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1000))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(num_labels)) 
model.add(Activation("sigmoid"))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="mean_squared_error",optimizer=adam,
              metrics=['accuracy']) 

#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#print(model.summary())

#Load weights if they exist
if init_weights_file!=None:
	print("Restoring Weights")
	model.load_weights(init_weights_file)

#Train Model
model.fit(X_train, Y_train, batch_size=32, epochs=num_epochs, verbose=1) 

print("Trained after " + str(time.time()-start_time))

#Evaluate Model
y_probs = model.predict_proba(X_train)							#Predicted probability that each label is positive
AU_PRC = average_precision_score(Y_train, y_probs)				#Area under PR curve taken as average of precisions for all recalls
precision, recall, _ = precision_recall_curve(Y_train, y_probs)	#Arrays for plotting precision recall curve
print('Max prob Train:', np.amax(y_probs))
print('Min prob Train:', np.amin(y_probs))
print('Train AUPRC:', AU_PRC)


y_probs = model.predict_proba(X_val)							#Predicted probability that each label is positive
AU_PRC = average_precision_score(Y_val, y_probs)				#Area under PR curve taken as average of precisions for all recalls
precision, recall, _ = precision_recall_curve(Y_val, y_probs)	#Arrays for plotting precision recall curve
print('Max prob val:', np.amax(y_probs))
print('Min prob val:', np.amin(y_probs))
print('Val AUPRC:', AU_PRC)
#score = model.evaluate(X_train, Y_train, verbose=0)
#print('Train loss:', score[0])
#print('Train accuracy:', score[1])
score = model.evaluate(X_val, Y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

#print(precision)
#print(recall)





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



print("Finished after " + str(time.time()-start_time))


#Display plots if they exist
#plt.show()


