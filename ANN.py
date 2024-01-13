import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your features (sj)
import features
sj = features.iq

# Setting the value for X and Y
x = sj[['weekofyear', 'reanalysis_dew_point_temp_k', 'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'reanalysis_air_temp_k', 'station_max_temp_c', 'reanalysis_min_air_temp_k', 'reanalysis_air_temp_k']]
y = sj['total_cases']

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

best_mae = np.inf
best_units = None
best_layers = None
best_batch_size = None
best_r_squared = None
best_mse = None

# Define ranges for hyperparameters
units_range = [32, 64, 128]
layers_range = [1, 2, 3]
batch_size_range = [16, 32, 64]

for units in units_range:
    for layers in layers_range:
        for batch_size in batch_size_range:
            # Building the neural network model
            model = Sequential()
            model.add(Dense(units=units, activation='relu', input_dim=x_train_scaled.shape[1]))

            for _ in range(layers - 1):
                model.add(Dense(units=units, activation='relu'))

            model.add(Dense(units=1, activation='linear'))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Fitting the model
            model.fit(x_train_scaled, y_train, epochs=50, batch_size=batch_size, validation_split=0.1, verbose=0)

            # Prediction on the test set
            y_pred_nn = model.predict(x_test_scaled).flatten()

            # Model Evaluation
            r_squared_nn = metrics.r2_score(y_test, y_pred_nn)
            mae_nn = metrics.mean_absolute_error(y_test, y_pred_nn)
            mse_nn = metrics.mean_squared_error(y_test, y_pred_nn)

            print(f'Units: {units}, Layers: {layers}, Batch Size: {batch_size}, R squared: {r_squared_nn}, MAE: {mae_nn}, MSE: {mse_nn}')

            # Check if the current configuration is the best in terms of MAE
            if mae_nn < best_mae:
                best_mae = mae_nn
                best_units = units
                best_layers = layers
                best_batch_size = batch_size
                best_r_squared = r_squared_nn
                best_mse = mse_nn

# Print the best configuration
print(f'Best Configuration - Units: {best_units}, Layers: {best_layers}, Batch Size: {best_batch_size}, Best R squared: {best_r_squared}, Best MAE: {best_mae}, Best MSE: {best_mse}')