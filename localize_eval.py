import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import adam
from keras.optimizers import sgd
import time
import json
import os 

import matplotlib
from matplotlib import pyplot as plt

cwd = os.getcwd()
test = np.load(cwd + "/datasets/synth_set_test.npy")
test_labels = test[:,0:3] # x,y,theta
test_data = test[:,3:] # depth scan 
m = np.shape(test_data)[0]


hidden_size = 300
model = Sequential()
model.add(Dense(hidden_size, input_shape=(np.shape(test_data)[1],), activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(np.shape(test_labels)[1], activation='linear')) 
model.compile(loss="mse", optimizer=adam()) # since no params given to adam means its using all paper defaults
model_name_to_load = cwd + "/models/base_loc_model.h5"
model.load_weights(model_name_to_load)



# get some live matplot lib plotting of real and predicted on the test set
plt.ion()
im = plt.imread("map.png")
for ex in range(0,100,5):
#    plt.clf() # Clear all figures
    implot = plt.imshow(im,origin='upper',extent=[-40,2,-2,30])

#    circle_real = plt.Circle((test_labels[ex,0],test_labels[ex,1]),1,color='r') # x y might be swapped 

#    print('in shape ',np.shape(test_data[ex:ex+1,:]))
    predictions = model.predict(test_data[ex:ex+1,:])
#    print(predictions.shape)
#    circle_pred = plt.Circle((predictions[0,0],predictions[0,1]),1,color='g') # x y might be swapped 

#    plt.gcf().gca().add_artist(circle_real)
#    plt.gcf().gca().add_artist(circle_pred)
#    plt.axis('equal')
    plt.scatter([test_labels[ex,1]],[test_labels[ex,0]],color='r')
    plt.scatter([predictions[0,1]],[predictions[0,0]],color='b')
    plt.xlim([-40,2])
    plt.ylim([-2,28])


#    plt.show()
    print(test_labels[ex,2],predictions[0,2])
#    print(test_labels[ex,:], predictions)
#    time.sleep(0.1)
    plt.pause( 0.01 )

plt.show()
#input()

    
