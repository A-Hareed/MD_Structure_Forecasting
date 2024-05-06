import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



data_bb = np.load('data_bb.npy')
data_aa = np.load('change_aa.npy')
torsion_angle = np.load('torsion_angles.npy')
com_Evector = np.load('com_e_vector_Test.npy')

print('tareget and feature shape',data_aa.shape,data_bb.shape,torsion_angle.shape,com_Evector.shape)

train_feature_bb = data_bb.reshape(-1,3)
tor_train = torsion_angle[1000:1500,:].reshape(-1,1)

com_train  = com_Evector.reshape(-1,1)


train_feature_bb = np.concatenate((train_feature_bb,tor_train,com_train),axis=1)



print('training and validation feature shape: ',train_feature_bb.shape)




def get_labels(arr):
    start = -12
    for i in range(0,arr.shape[1],12):
        start = i
        end  = i+12
        print(start,end)
        if i ==0:
            label = arr[:,start:end]
        else:
            label = np.concatenate((label,arr[:,start:end]),axis=0)
    
    return label





# train_label = get_labels(data_aa[:30,:])
# val_label = get_labels(data_aa[30:40,:])
# print('train and valdiation shapes',train_label.shape, val_label.shape)
# print(train_label[:10])


# Standardization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_aa)



train_label = get_labels(scaled_data[1000:1500,:])

print('train and valdiation shapes',train_label.shape)
print(train_label)
print('train and valdiation shapes',train_label.max(), train_label.min())


print('training starts: ')


Deep_model = keras.models.load_model('model.h5')

print(Deep_model.summary())
# print(Deep_model.evaluate(train_feature_bb,train_label,batch_size=1000))

yhat = Deep_model.predict(train_feature_bb)


print(yhat)
print(yhat.max(),yhat.min())
