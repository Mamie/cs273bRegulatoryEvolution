
np.random.seed(1234)
import keras;
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
from keras.constraints import maxnorm;
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, activity_l1, activity_l2


seq_length = 2000 #2kb iput
num_labels = 2    #Our binary final prediction
##########################
init_weights_file = None #load in pre-trained weights Anna gave us here if we're using them. Elsewise, leave as None.
##########################

'''
Soft Parameter Sharing
Weight regularizers can be custom-made as per: https://keras.io/regularizers/
Weights (or at least their values) can be accessed as per: https://github.com/fchollet/keras/issues/91 , https://github.com/fchollet/keras/issues/4209 and https://stackoverflow.com/questions/43305891/how-to-correctly-get-layer-weights-from-conv2d-in-keras
At this point, it isn't clear to me how to get loss to include a term for dissimilarity between parameters in different models.

My best attempt at getting to work is as follows:

def sq_dist(w_A):
	w_B = #load in already trained weights from species B
	return tf.reduce_mean(tf.squared_difference(w_A, w_B)) #return squared difference of the weight matrices for the two species

This would require kernel_regularizer=sq_dist to be added as an argument to each layer whose weights we want linked to those of another species

Not sure if this would actually work
'''


model_species_A = Sequential() #Multiple models can be made, each corresponding to a different species
model_species_A.add(Convolution2D(300,4,19,input_shape=(1,4,int(seq_length)), W_learning_rate_multiplier=10.0))	
model_species_A.add(PReLU())
model_species_A.add(MaxPooling2D(pool_size=(1,3)))

model_species_A.add(Convolution2D(200,1,11,W_learning_rate_multiplier=5.0,W_constraint=maxnorm(m=7)))
model_species_A.add(BatchNormalization(mode=0, axis=1))
model_species_A.add(PReLU())
model_species_A.add(MaxPooling2D(pool_size=(1,4)))

model_species_A.add(Convolution2D(200,1,7,W_constraint=maxnorm(m=7)))
model_species_A.add(BatchNormalization(mode=0, axis=1))
model_species_A.add(PReLU())
model_species_A.add(MaxPooling2D(pool_size=(1,4)))

model_species_A.add(Flatten())
model_species_A.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
model_species_A.add(PReLU())
model_species_A.add(Dropout(0.3))

model_species_A.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
model_species_A.add(PReLU())
model_species_A.add(Dropout(0.3))

model_species_A.add(Dense(num_labels)) 
model_species_A.add(Activation("sigmoid"))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_species_A.compile(loss="binary_crossentropy",optimizer=adam, metrics=['accuracy']) #This may be where loss can be changed to incentivise weights to be similar to weights for other species

#model_species_A.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1) #Training the model_species_A. X_train and Y_train are our training inputs and labels
#score = model_species_A.evaluate(X_test, Y_test, verbose=0)

if init_weights_file!=None:
    model_species_A.save_weights(init_weights_file)		


