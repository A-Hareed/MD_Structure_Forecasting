# Imports:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# import tensorflow for machine learning
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
# from tensorflow.keras.utils import timeseries_dataset_from_array


# load normalised training/validation Data *************************************************
train_X = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_x_cleaned.npy')
train_Y = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_y_cleaned.npy')
val_X = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_vx_cleaned.npy')
val_Y = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_vy_cleaned.npy')





# reshape training/validation data so that each 
# sequence represents atomic coordinates 
# new shape (atom, time-frames, [X,Y,Z])

 
sequence_length = 50
features_len = 3

# train_Y = train_Y.reshape(51148,sequence_length*features_len)
# val_Y =  val_Y.reshape(24228,sequence_length*features_len)
train_Y = train_Y.reshape(104988,sequence_length*features_len)
val_Y =  val_Y.reshape(51148,sequence_length*features_len)


# build RNN **************************************************************************

inputs = Input(shape=(sequence_length,features_len)) # input shape equal to three features in given sequence

# First LSTM layer, consisting of 64 neurons and since it connects to another LSTM 
# return_sequences equal true
x = layers.LSTM(64,recurrent_dropout=0.2,return_sequences=True,name='RNN_1')(inputs)
x = layers.Dropout(0.2)(x)
# second LSTM layer. doesn't return sequence since it connected to a Dense layer
x = layers.LSTM(128,name='RNN_2')(x)
x = layers.Dropout(0.2)(x)
# output layer has neurons equal to 3 (coordinate feature) multiplied by 
# number of time-frame 
output = Dense(sequence_length*features_len,name='Dense_layer_output')(x) 
model_RNN = keras.Model(inputs,output)


print(model_RNN.summary())

# compile the RNN model 
model_RNN.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0015), loss='mse',metrics=['mae',keras.metrics.RootMeanSquaredError()])

# train the model
history_rnn = model_RNN.fit(train_X , train_Y,
            epochs=8,
            batch_size = 2692,
            validation_data=(val_X, val_Y )
            )

# load testing datasets 
# Testing data set in different shape than training/validation  testing shape (1, time-frame, 8076) 
# where all coordinates of the protein is the shape of the last dim 
test_X = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_testx_cleaned.npy')
test_Y = np.load('/Users/user/Documents/MDForecasting/sims_forces/coo1_testy_cleaned.npy')
test_Y =  test_Y #.reshape(24228,sequence_length*features_len)

print(f'testing shape{test_X.shape}')

# reshape to (atom, time-frames, [X,Y,Z]) for ML input
for i in list(range(0,test_X.shape[-1],3)):

    print(test_X[0,:,i:(i+3)].shape)

    print(i)
    if i == 0:
        new_arr = test_X[0,:,i:(i+3)].reshape(1,50,3)

    else:
        new_arr = np.append(new_arr,test_X[0,:,i:(i+3)].reshape(1,50,3),axis=0)





# get scaler 
scaler = joblib.load('/Users/user/Documents/MDForecasting/sims_forces/scaler_coo.save')



# Make predictions:
for i in range(4):
    print(f'Time Step: {i+1}')
    if i == 0:
        # ex = model_RNN.predict(new_arr[0].reshape(1,100,3))
        ex = model_RNN.predict(new_arr)
        # ex_reshaped = ex.reshape(100,3)
        ex_reshaped = ex.reshape(2692,50,3)
        
        # add the part that reshapes the dataFrame ############
        k = 0

        for i in new_arr:
            # print(i.shape)

            if k == 0:
                num_arr = i
                
            else: 

                num_arr = np.append(num_arr,i,axis=1)
                


            k +=1
        

        df = pd.DataFrame(num_arr)


    else:
        # ex = model_RNN.predict(new_arr[0].reshape(1,100,3))
        ex = model_RNN.predict(ex_reshaped.reshape(2692,50,3))
        # ex_reshaped = ex.reshape(100,3)
        ex_reshaped = ex.reshape(2692,50,3)
        
        # add the part that reshapes the dataFrame ############
        k = 0

        for i in new_arr:
            # print(i.shape)

            if k == 0:
                num_arr = i
                
            else: 

                num_arr = np.append(num_arr,i,axis=1)
           


            k +=1



        df_temp = pd.DataFrame(num_arr)

        df = pd.concat([df,df_temp],ignore_index=True)



df_scaled = pd.DataFrame(scaler.inverse_transform(df))
print('scalled dataframe ')
print(df_scaled.head())


#**************************************************************************
# get PDB files
from Bio.PDB import PDBParser, PDBIO
io = PDBIO()
p = PDBParser()

def generate_sims(start_arr): #,MODEL,p_steps,SCALER):
    counter = 0 
    # create pdb parsers
    io = PDBIO()
    p = PDBParser()
    structure = p.get_structure("2lhi", "/Users/user/Documents/MDForecasting/pdb_example.pdb")


    for arr in start_arr:
        counter +=1
        
        
        arr_final = []
        arr_temp = []
        k = 0

        for i in arr:
            k +=1
            if k <4:
                arr_temp.append(i)

            if k ==3:
                arr_final.append(arr_temp)
                arr_temp = []
                k =0


        i = 0
        for model in structure:

            for chain in model:
                for residue in chain: 
                    for atom in residue:
                        atom.set_coord(arr_final[i])
                        # anum = atom.get_serial_number() #code added
                        # atom.set_serial_number(anum)    #code added
                        i += 1
        
        
        io.set_structure(structure)
        io.save(f'/Users/user/Documents/MDForecasting/sims_forces/OutPut_Lstm.pdb', preserve_atom_numbering = True)



        # check if file exist:
        path = '/Users/user/Documents/MDForecasting/sims_forces/full_output_200_pred_seq_len_50.pdb'
   
        # Check whether the specified
        # path exists or not
        isExist = os.path.exists(path)
        if isExist == False:
            with open(path,'w') as fp:
                data = f'MODEL        {counter}\n'
                fp.write(data)
            with open('/Users/user/Documents/MDForecasting/sims_forces/OutPut_Lstm.pdb') as fp:
                data2 = fp.read()

            with open(path,'a') as fp:
                fp.write(data2)

        else:
            
            # Reading data from file2
            with open('/Users/user/Documents/MDForecasting/sims_forces/OutPut_Lstm.pdb') as fp:

                data2 = fp.read()

            with open(path,'a') as fp:
                data = f'MODEL        {counter}\n'
                fp.write(data)
                fp.write(data2)




generate_sims(df_scaled.to_numpy())

exit()

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



# use -pbc nojump or -pbc cluster



# https://manual.gromacs.org/documentation/current/onlinehelp/gmx-trjconv.html

