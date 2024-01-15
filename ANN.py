import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#import features
#sj = features.iq

#x = sj[['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'station_min_temp_c']]
#y = sj['total_cases']

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

#scaler = StandardScaler()
#x_train_scaled = scaler.fit_transform(x_train)
#x_test_scaled = scaler.transform(x_test)

#best_mae = np.inf
#best_units = None
#best_layers = None
#best_batch_size = None
#best_r_squared = None
#best_mse = None

#units_range = [32, 64, 128]
#layers_range = [1, 2, 3]
#batch_size_range = [16, 32, 64]

#history_dict = {}

#for units in units_range:
#    for layers in layers_range:
#        for batch_size in batch_size_range:
#           
#            model = Sequential()
#            model.add(Dense(units=units, activation='relu', input_dim=x_train_scaled.shape[1]))

 #           for _ in range(layers - 1):
 #               model.add(Dense(units=units, activation='relu'))

#            model.add(Dense(units=1, activation='linear'))
           
#            model.compile(optimizer='adam', loss='mean_squared_error')
#          
#            history = model.fit(x_train_scaled, y_train, epochs=60, batch_size=batch_size, validation_split=0.1, verbose=0)
#            history_dict[(units, layers, batch_size)] = history.history       
#           
#           y_pred_nn = model.predict(x_test_scaled).flatten()
#
#           r_squared_nn = metrics.r2_score(y_test, y_pred_nn)
#           mae_nn = metrics.mean_absolute_error(y_test, y_pred_nn)
#           mse_nn = metrics.mean_squared_error(y_test, y_pred_nn)
#
#           print(f'Units: {units}, Layers: {layers}, Batch Size: {batch_size}, R squared: {r_squared_nn}, MAE: {mae_nn}, MSE: {mse_nn}')
#
#           if mae_nn < best_mae:
#                best_mae = mae_nn
#                best_units = units
#                best_layers = layers
#                best_batch_size = batch_size
#                best_r_squared = r_squared_nn
#                best_mse = mse_nn

#print(f'Best Configuration - Units: {best_units}, Layers: {best_layers}, Batch Size: {best_batch_size}, Best R squared: {best_r_squared}, Best MAE: {best_mae}, Best MSE: {best_mse}')

#for config, history in history_dict.items():
#    plt.plot(history['loss'])
#    plt.plot(history['val_loss'])
   
#plt.title('Training and Validation Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

def examine(y_pred, y_test):
    c = pd.DataFrame({
        'y' : y_test,
        'p' : y_pred
     })
    c = c.sort_index()
    plt.figure(figsize=[15,4])
    plt.plot(c.y, color='green')
    plt.plot(c.p, color='red')
    plt.show()

import features
sj = features.sj
iq = features.iq

def run_neural_network(x, y, city_name):
    x = x[['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'station_min_temp_c']]
    y = y['total_cases']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    units = 32
    layers = 1
    batch_size = 64

    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_dim=x_train_scaled.shape[1]))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))

    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_scaled, y_train, epochs=60, batch_size=batch_size, validation_split=0.1, verbose=0)

    y_pred_nn = model.predict(x_test_scaled).flatten()

    r_squared_nn = metrics.r2_score(y_test, y_pred_nn)
    mae_nn = metrics.mean_absolute_error(y_test, y_pred_nn)
    mse_nn = metrics.mean_squared_error(y_test, y_pred_nn)
    
    #examine(y_pred_nn,y_test)

    print(f'{city_name} - R squared: {r_squared_nn}, MAE: {mae_nn}, MSE: {mse_nn}')

    return model


sj_model = run_neural_network(sj, sj, 'sj')
iq_model = run_neural_network(iq, iq, 'iq')

test_data = pd.read_csv('data/dengue_features_test.csv')

sj_actualtest = test_data[test_data.city == 'sj'].copy()
iq_actualtest = test_data[test_data.city == 'iq'].copy()

sj_actualtest = sj_actualtest.interpolate(method='linear', limit_direction='forward')
iq_actualtest = iq_actualtest.interpolate(method='linear', limit_direction='forward')

sj_actualtest = sj_actualtest[['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'station_min_temp_c']]
iq_actualtest = iq_actualtest[['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_avg_temp_c', 'station_min_temp_c']]

scaler = StandardScaler()
sj_actualtest_scaled = scaler.fit_transform(sj_actualtest)
iq_actualtest_scaled = scaler.transform(iq_actualtest)

sj_predictions = sj_model.predict(sj_actualtest_scaled).flatten()
iq_predictions = iq_model.predict(iq_actualtest_scaled).flatten()

sj_predictions = np.round(sj_predictions).astype(int)
iq_predictions = np.round(iq_predictions).astype(int)

all_predictions = np.concatenate((sj_predictions, iq_predictions))

#print(all_predictions)

submission = pd.read_csv('data/submission_format.csv',index_col=[0, 1, 2])
submission.total_cases = all_predictions
submission.to_csv("submissions/nn.csv")
