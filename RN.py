# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
import datetime

data=pd.read_excel('datos.xls')

X = data.iloc[:,:31]
X=X.values
y =data.iloc[:,32]
y=y.values

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
SimpleImputer()
X = imp.transform(X)

X_normalized = preprocessing.normalize(X, norm='l2')

train_data, test_data, train_targets, test_targets = train_test_split(X_normalized, y, test_size=0.1)

def get_regularised_model(wd, rate):
    model = Sequential([
        Dense(32, kernel_regularizer=regularizers.l2(wd), 
              kernel_initializer='he_uniform', bias_initializer='ones',
              activation="relu", input_shape=(train_data.shape[1],)),
        
        BatchNormalization(),  # <- Batch normalisation layer
        Dropout(rate),
        BatchNormalization(),  # <- Batch normalisation layer
    
        Dense(128/2, kernel_regularizer=regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(1, activation='softmax')
    ])
    return model 

model = get_regularised_model(1e-5, 0.1)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse','accuracy'])

# log_dir = "logs2/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_data, train_targets, epochs=100, 
                    validation_split=0.15, batch_size=64, verbose=2)

# model.fit(x=train_data, 
#           y=train_targets, 
#           epochs=5, 
#           validation_split=0.15, 
#           callbacks=[tensorboard_callback])

print('\n Evaluate')
model.evaluate(test_data, test_targets, verbose=2)
               
print('\n Predict')               
model.predict(test_data, verbose=2)           