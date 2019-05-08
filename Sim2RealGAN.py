import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import adam
from keras.callbacks import History 
from keras.callbacks import EarlyStopping
from keras import metrics
import os 
history = History()

import time
import json


import matplotlib
from matplotlib import pyplot as plt
print("Using:",matplotlib.get_backend())

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

fakes = np.load(cwd + "/datasets/mini_synth_set.npy")
fakes_data = fakes[:,3:] # depth scan
fake_data = fakes_data 
fakes_labels = np.zeros((np.shape(fakes)[0],1))
fakes_data = np.concatenate((fakes_labels,fakes_data),1)

reals_data = np.load(cwd + "/datasets/mini_real_set.npy")
real_data = reals_data
reals_labels = np.ones((np.shape(reals_data)[0],1))
reals_data = np.concatenate((reals_labels,reals_data),1)

# even data 
if np.shape(fakes)[0] >= np.shape(reals_data)[0]:
    train_data = np.concatenate((fakes_data[0:np.shape(reals_data)[0],:],reals_data),0)
else:
    train_data = np.concatenate((fakes_data,reals_data[0:np.shape(fakes)[0],:]),0)

print(train_data.shape)
np.random.shuffle(train_data) # shuffles the dataset row wise
train_labels = train_data[:,0]
train_data = train_data[:,1:]


hidden_size = 300
filters = 10
kernel_size = 5
pool_size = 3

Gen_model = Sequential()
Gen_model.add(Conv1D(filters,kernel_size,padding='same',activation='relu', input_shape=(np.shape(real_data)[1],1)))
Gen_model.add(Conv1D(filters,kernel_size,padding='same',activation='relu'))
Gen_model.add(Conv1D(filters,kernel_size,padding='same',activation='relu'))
#Gen_model.add(Conv1D(1,kernel_size,padding='same',activation='relu'))
#Gen_model.add(Dense(np.shape(real_data)[1], activation='relu'))
#Gen_model.add(Dense(np.shape(real_data)[1], activation='relu'))
#Gen_model.add(Dense(np.shape(real_data)[1], activation='relu'))
Gen_model.add(Conv1D(1,kernel_size,padding='same',activation='relu'))
Gen_model.compile(loss="binary_crossentropy", optimizer=adam()) # since no params given to adam means its using all paper defaults

Dis_model = Sequential()
Dis_model.add(Dense(hidden_size, input_shape=(np.shape(real_data)[1],), activation='relu'))
Dis_model.add(Dense(hidden_size, activation='relu'))
Dis_model.add(Dense(hidden_size, activation='relu'))
Dis_model.add(Dense(hidden_size, activation='relu'))
Dis_model.add(Dense(1, activation='sigmoid')) 
Dis_model.compile(loss="binary_crossentropy", optimizer=adam(),metrics=['accuracy']) # since no params given to adam means its using all paper defaults

Gen_model_name_to_load = cwd + "/models/Gen_model.h5"
Gen_model_name_to_save = cwd + "/models/Gen_model.h5"

Dis_model_name_to_load = cwd + "/models/Dis_model.h5"
Dis_model_name_to_save = cwd + "/models/Dis_model.h5"


# If you want to continue training from a previous model, just uncomment the line bellow
#Gen_model.load_weights(Gen_model_name_to_load)
#Dis_model.load_weights(Dis_model_name_to_load)

patience = 4
Dis_loss = []
Gen_loss = []
cus_epochs = 275
train_Dis = 0 # T/F
train_Gen = 1 # T/F
for e in range(cus_epochs):

    if train_Dis:
        # use the generator and train the discriminator
        # generate some fakes
#        fake_data += np.random.normal(0,(fake_data.shape[0],fake_data.shape[1]))
        gen_images = Gen_model.predict(np.reshape(fake_data,(fake_data.shape[0],fake_data.shape[1],1))) 

        X = gen_images
        batc = real_data[0:np.shape(X)[0],:]
        X = np.concatenate((X,np.reshape(batc,(batc.shape[0],batc.shape[1],1))),0)
        Y = np.concatenate((np.zeros((gen_images.shape[0],1,1)),np.ones((batc.shape[0],1,1))),0)
        temp = np.concatenate((Y,X),1)
        np.random.shuffle(temp)
        X = temp[:,1:]
        X = np.reshape(X,(X.shape[0],X.shape[1]))
        Y = temp[:,0]
        
        loss, acc = Dis_model.train_on_batch(X,Y)
        Dis_loss.append(loss)
        print('Training discriminator... ','epoch: ',e,' loss: ',loss,' acc: ',acc)

        # custom early stopping
        if len(Dis_loss) > patience + 1:
            if not np.any(Dis_loss[-1] >= Dis_loss[-patience:]): break

    if train_Gen:
        batch = np.reshape(fake_data,(fake_data.shape[0],fake_data.shape[1],1))
        gen_images = Gen_model.predict(batch) 
#        print('gen shape:',gen_images.shape)
        Y = Dis_model.predict(np.reshape(gen_images,(gen_images.shape[0],gen_images.shape[1])))
#        Y = 1-Y
        D_acc = np.mean(Y)
        Y = np.reshape(Y,(Y.shape[0],1,1))
#        print('Y shape: ',Y.shape)
        Y = np.repeat(Y,fake_data.shape[1],axis = 1)
#        print('Y shpae: ',Y.shape)
        loss = Gen_model.train_on_batch(batch,Y)
        Gen_loss.append(loss)
        print('Training generator... ','epoch: ',e,' loss: ',loss,' D accuracy: ',D_acc)

        # custom early stopping
        if len(Gen_loss) > patience + 1 :
            if not np.any(Gen_loss[-1] >= Gen_loss[-patience:]): break

# Save trained model weights and architecture

if train_Gen:
    plt.plot(Gen_loss)
    plt.title('Generator loss')
    Gen_model.save_weights(Gen_model_name_to_save, overwrite=True)
    with open("Gen_model.json", "w") as outfile:
        json.dump(Gen_model.to_json(), outfile)
if train_Dis:
    plt.plot(Dis_loss)
    plt.title('Discriminator loss')
    Dis_model.save_weights(Dis_model_name_to_save, overwrite=True)
    with open("Dis_model.json", "w") as outfile:
        json.dump(Dis_model.to_json(), outfile)

plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()
