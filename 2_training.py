import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers import LSTM
from keras import callbacks

startdir = 'flows/training-dataset'
flows_directory = 'flows/training-dataset'
dataframe_file = 'flows/dataframe.csv'
outcomes_file = 'flows/outcomes.csv'
model_file = 'models/'
scaler_file = 'scaler/scaler.save'
encoder_file = 'encoder/encoder.save'


# Reading dataframe
df = pd.read_csv(dataframe_file)
outcomes = pd.read_csv(outcomes_file)

print('dataframe shape is', df.shape)


x_train, x_test, y_train, y_test = train_test_split(df, outcomes)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


# create the model
model = Sequential()
model.add(LSTM(256,input_shape=(1,x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=128, callbacks =[earlystopping])

model.save(model_file)
print('Saved model to disk')