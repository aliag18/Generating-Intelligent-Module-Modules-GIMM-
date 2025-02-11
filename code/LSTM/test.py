import sys
 
# setting path
sys.path.append('../../LSTM')
from data import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,  Flatten

import matplotlib.pyplot as plt

from statistics import mean, stdev
import pandas as pd
import numpy as np
import datetime
import csv

# how to run model vs how to use model as a callable function


start = datetime.datetime.now()

past_hours = 6
n_steps = past_hours
next_hours = 4
next_steps = next_hours

transformations = [e for n in select_stations for e in ('tmp'+str(n), 'swrad'+str(n), 'lwrad'+str(n), 'wind'+str(n))]
print(transformations)

transformations = [(str(i), MinMaxScaler(copy=True), make_column_selector(pattern = e)) for i,e in enumerate(transformations)]
# <sklearn.compose._column_transformer.make_column_selector object at 0x7fef781b29b0>
# for i,e in enumerate(transformations):
#     print(str(i))
#     print(e)
# print(transformations)

transformations += [(str(len(transformations)),OneHotEncoder(), make_column_selector(pattern = 'hour'))]


transformations += [(str(len(transformations)),OneHotEncoder(), make_column_selector(pattern = 'day'))]


transformations += [(str(len(transformations)),OneHotEncoder(), make_column_selector(pattern = 'month'))]


transformations += [("y",StandardScaler(copy=True), make_column_selector(pattern = 'ERCOT'))]


ct = ColumnTransformer(transformations)
# print(ct)

# normed_data = pd.DataFrame(ct.fit_transform(load_data)).astype(np.float32)
# Transform the data with the ColumnTransformer
normed_data = ct.fit_transform(load_data)

# Convert the sparse matrix to a dense array and then to a DataFrame
normed_data = pd.DataFrame(normed_data.toarray()).astype(np.float32)

dataset = normed_data.values
print(dataset)

dataset = normed_data.values
steps = n_steps+next_steps
time_series =  np.array([dataset[i:i + steps].copy() for i in range(len(dataset) - steps) if load_data.index[i+steps] - load_data.index[i] == datetime.timedelta(hours=steps)])

print(time_series.shape)

X = time_series[:, :n_steps, :]
Y = time_series[:, n_steps:, -1]

print(X.shape, Y.shape)

#sizes for test, train, and validation datasets
a = len(time_series)
b = int(0.9*a)
c = int(a-b)//2

X_train,    Y_train =   X[:b],      Y[:b]
X_val,      Y_val   =   X[b:b+c],   Y[b:b+c]
X_test,     Y_test  =   X[b+c:],    Y[b+c:]

print(X_train.shape, X_val.shape, X_test.shape)
print(Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])),
    LSTM(256, return_sequences=True),
    Flatten(),
    Dense(128, activation="sigmoid"),
    Dropout(0.1),
    Dense(128, activation="sigmoid"),
    Dropout(0.1),
    Dense(128, activation="sigmoid"),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(next_steps)
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(5e-4,
                                                decay_steps=1000000,
                                                decay_rate=0.98,
                                                  staircase=False)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=1e-6)

model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
model.summary()

EPOCHS = 10

print(type(X_val))  # Should be <class 'numpy.ndarray'>
print(type(Y_val))  

# print(np.isnan(X_train).any())  # Check for NaNs
# print(np.isfinite(X_train).all()) 
print(f"X_train dtype: {X_train.dtype}")
print(f"Y_train dtype: {Y_train.dtype}")

history = model.fit(x = X_train, y = Y_train, epochs=EPOCHS, validation_data = (X_val, Y_val), shuffle=False, verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test_predictions = model.predict(X_test)
op_list= [e for y in test_predictions for e in y]   #predicted list
ip_list = [e for y in Y_test for e in y]    #actual list
print(len(op_list), len(ip_list))

test_labels = pd.DataFrame({'actual':ip_list})
predictions = pd.DataFrame({'predict':op_list})
test_labels['actual'] = ct.named_transformers_['y'].inverse_transform(test_labels)
predictions['predict'] = ct.named_transformers_['y'].inverse_transform(predictions)

error = predictions['predict'] - test_labels['actual']
rerror = 100*(predictions['predict'] - test_labels['actual'])/test_labels['actual']

print('Mean of error in test data:', mean(error))
print('Mean of absolute error in test data:', mean(abs(error)))
print('Root of mean of squared error in test data:', mean(error**2)**0.5)
print('Standard deviation of error in test data:', stdev(error))
print("=============================================================")
print('MPE (%):', mean(rerror))
print('MAPE (%):', mean(abs(rerror)))
print('RMSPE (%):', mean(rerror**2)**0.5)
print('Deviation of error (%):', stdev(rerror))
print("=============================================================")
print('Mean of test data:', mean(test_labels['actual']))
print('Standard deviation of test data:', stdev(test_labels['actual']))
print("Coefficient of determination:",r2_score(test_labels['actual'], predictions['predict']))

#absolute errors
for threshold in [5,10,50,100,500,1000,1500,2000,3000,4000,5000]:
    accuracy = sum(abs(e) < threshold for e in error)/len(error)*100
    print('accuracy:', accuracy,'% with threshold of', threshold)

#relative errors
for threshold in [0.5,1,1.5,2,3,4,5]:
    accuracy = sum(abs(e) < threshold for e in rerror)/len(error)*100
    print('accuracy:', accuracy,'% with threshold of', threshold)

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

plt.figure(figsize=(5,5))
plt.hist(error, bins = 50)
plt.xlabel("Prediction Absolute Error")
_ = plt.ylabel("Count")

plt.figure(figsize=(5,5))
plt.hist(rerror, bins = 50)
plt.xlabel("Prediction Error (%)")
_ = plt.ylabel("Count")

plt.figure(figsize=(10,5))
a = plt.axes(aspect='equal')
plt.scatter(test_labels['actual'], predictions)
plt.xlabel('True Value')
plt.ylabel('Predictions')
lims = [25000, 80000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

end = datetime.datetime.now()
print('total time:', end-start)

from keras.models import save_model
save_model(model,'Models/LSTM_model_3.0.1.h5',include_optimizer=True)
np.save('Models/LSTM_history_3.0.1.npy',history.history)

import numpy as np
history = np.load('Models/LSTM_history_3.0.1.npy', allow_pickle=True)[()]
print(history)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

plt.figure(figsize=(5,5))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
# plt.title('model loss')
plt.ylabel('model loss')
plt.xlabel('epoch number')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

from keras.models import load_model
custom_objects = {'mse' : 'mean_squared_error'}
model = load_model('Models/LSTM_model_3.0.1.h5', custom_objects=custom_objects)

np.save('Models/predictions1.0', np.array(predictions['predict']))

# for i in range(0, len(normed_data)):
#     print(normed_data(i))
# print(normed_data)

# # Write CSV file
# with open("test.csv", "wt") as fp:
#     writer = csv.writer(fp, delimiter=",")
#     # writer.writerow(["your", "header", "foo"])  # write header
#     writer.writerows(dataset)