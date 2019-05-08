import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import adam
from keras.optimizers import sgd
from keras.callbacks import History 
from keras.callbacks import EarlyStopping
history = History()

import time
import json


import matplotlib
from matplotlib import pyplot as plt
print("Using:",matplotlib.get_backend())

import os 
cwd = os.getcwd()

## saving and setting up datasets
#test_data = np.load('mini_synth_set.npy')
#test_data = np.load('synth_set.npy')
#np.random.shuffle(test_data) # shuffles the dataset row wise
#num_train = int(np.shape(test_data)[0]*0.8)
#num_val = int(np.shape(test_data)[0]*0.1)
#num_test = int(np.shape(test_data)[0]*0.1)

#train = test_data[0:num_train,:]
#test = test_data[num_train:num_train+num_test,:]
#val = test_data[num_train+num_test:num_train+num_test+num_val,:]

#np.save('synth_set_train.npy',train)
#np.save('synth_set_val.npy',val)
#np.save('synth_set_test.npy',test)

train = np.load('synth_set_train.npy')
val = np.load('synth_set_val.npy')
test = np.load('synth_set_test.npy')

train_labels = train[:,0:3] # x,y,theta
train_data = train[:,3:] # depth scan 

val_labels = val[:,0:3] # x,y,theta
val_data = val[:,3:] # depth scan 

test_labels = test[:,0:3] # x,y,theta
test_data = test[:,3:] # depth scan 


hidden_size = 300

model = Sequential()
model.add(Dense(hidden_size, input_shape=(np.shape(train_data)[1],), activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(np.shape(train_labels)[1], activation='linear')) 
model.compile(loss="mse", optimizer=adam()) # since no params given to adam means its using all paper defaults

model_name_to_load = "base_loc_model.h5"
model_name_to_save = "base_loc_model.h5"

# If you want to continue training from a previous model, just uncomment the line bellow
model.load_weights(model_name_to_load)

# setup early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

# Fit the model
#model.fit(train_data, train_labels, validation_split=0.1, epochs=10, batch_size=64, callbacks=[history])
# specify val data so can reload weights and data and not have leaks from train to val
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=2, batch_size=64, callbacks=[history]) 

#print(history.history.keys())
#print(history.history['loss'])

#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save trained model weights and architecture, this will be used by the visualization code
model.save_weights(model_name_to_save, overwrite=True)
with open("model.json", "w") as outfile:
    json.dump(model.to_json(), outfile)

plt.plot(history.history['loss'],'b',label='train loss')
plt.plot(history.history['val_loss'],'r',label='val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

