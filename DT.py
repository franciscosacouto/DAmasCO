import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Change to DecisionTreeRegressor for regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import features
from sklearn.preprocessing import MinMaxScaler


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

#Import of data
sj=features.sj
iq=features.iq
scaler = MinMaxScaler()

#Attributes used
attributes = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c', 'weekofyear', 'total_cases']

# Create a new DataFrame with the selected columns
df = sj[attributes]

#Division of training and test data
X_train, X_test, y_train, y_test = train_test_split(df.drop('total_cases', axis=1), df['total_cases'], test_size = 0.2, random_state=0)

#Normalization of the data
normalized_data = scaler.fit_transform(X_train)
X_train = pd.DataFrame(normalized_data, columns=X_train.columns)
normalized_data = scaler.fit_transform(X_test)
X_test = pd.DataFrame(normalized_data, columns=X_test.columns)

#Fitting the Decision Tree Model
clf = DecisionTreeRegressor(min_samples_split=5, max_depth=50,random_state=0)
clf.fit(X_train, y_train)

#Prediction of validation set
y_pred = clf.predict(X_test)

#Graphs
examine(y_pred,y_test)

#Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

#prediction model
attributes2 = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c', 'weekofyear' ]
rawfeats = pd.read_csv('data/dengue_features_test.csv')
sj_validation = rawfeats[rawfeats.city=='sj'].copy()
sj_validation = sj_validation[attributes2]
sj_validation = sj_validation.interpolate(method='linear', limit_direction='forward')
sj_norm = scaler.fit_transform(sj_validation)
sj_validation = pd.DataFrame(sj_norm, columns=sj_validation.columns)
y_validation_dt_sj= clf.predict(sj_validation)
y_validation_dt_sj = y_validation_dt_sj.astype(int)



#---iq--
#Attributes used
attributes = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c', 'weekofyear',  'total_cases']

# Create a new DataFrame with the selected columns
df = iq[attributes]

#Division of training and test data
X_train, X_test, y_train, y_test = train_test_split(df.drop('total_cases', axis=1), df['total_cases'], test_size = 0.2, random_state=0)

#Normalization of the data
normalized_data = scaler.fit_transform(X_train)
X_train = pd.DataFrame(normalized_data, columns=X_train.columns)
normalized_data = scaler.fit_transform(X_test)
X_test = pd.DataFrame(normalized_data, columns=X_test.columns)

#Fitting the Decision Tree Model
clf = DecisionTreeRegressor(min_samples_split=5, max_depth=50,random_state=0)
clf.fit(X_train, y_train)

#Prediction of validation set
y_pred = clf.predict(X_test)

#Graphs
examine(y_pred,y_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

#prediction model
attributes2 = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c', 'weekofyear']
rawfeats = pd.read_csv('data/dengue_features_test.csv')
iq_validation = rawfeats[rawfeats.city=='iq'].copy()
iq_validation = iq_validation[attributes2]
iq_validation = iq_validation.interpolate(method='linear', limit_direction='forward')
iq_norm = scaler.fit_transform(iq_validation)
iq_validation = pd.DataFrame(iq_norm, columns=iq_validation.columns)
y_validation_dt_iq= clf.predict(iq_validation)
y_validation_dt_iq = y_validation_dt_iq.astype(int)

#submission
submission = pd.read_csv('data/submission_format.csv',index_col=[0, 1, 2])
submission.total_cases = np.concatenate([y_validation_dt_sj, y_validation_dt_iq])
submission.to_csv("submissions/dt.csv")