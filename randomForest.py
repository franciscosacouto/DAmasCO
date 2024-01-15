import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

sj=features.sj
iq= features.iq
scaler = MinMaxScaler()


features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c','weekofyear']

sj_train_x, sj_val_x, sj_train_y, sj_val_y = train_test_split(sj[features], sj['total_cases'], test_size=0.2, random_state=28)

normalized_data = scaler.fit_transform(sj_train_x)
sj_train_x = pd.DataFrame(normalized_data, columns=sj_train_x.columns)
normalized_data = scaler.fit_transform(sj_val_x)
sj_val_x = pd.DataFrame(normalized_data, columns=sj_val_x.columns)

regressor = RandomForestRegressor(max_depth=50,n_estimators=300, random_state =0,criterion='absolute_error')
regressor.fit(sj_train_x,sj_train_y)
train_pred = regressor.predict(sj_train_x)
val_pred = regressor.predict(sj_val_x)

mae=(mean_absolute_error(sj_train_y,train_pred))
print("MAE train:",mae)
mae=(mean_absolute_error(sj_val_y,val_pred))
print ("MAE validation:",mae)
examine(val_pred,sj_val_y)

rawfeats = pd.read_csv('data/dengue_features_test.csv')
sj_test = rawfeats[rawfeats.city=='sj'].copy()
sj_test = sj_test[features]
sj_test = sj_test.interpolate(method='linear', limit_direction='forward')
sj_norm = scaler.fit_transform(sj_test)
sj_test = pd.DataFrame(sj_norm, columns=sj_test.columns)
sj_test_preds = regressor.predict(sj_test)
sj_test_preds = sj_test_preds.astype(int)

features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c','weekofyear']


iq_train_x, iq_val_x, iq_train_y, iq_val_y = train_test_split(iq[features], iq['total_cases'], test_size=0.2, random_state=28)

normalized_data = scaler.fit_transform(iq_train_x)
iq_train_x = pd.DataFrame(normalized_data, columns=iq_train_x.columns)
normalized_data = scaler.fit_transform(iq_val_x)
iq_val_x = pd.DataFrame(normalized_data, columns=iq_val_x.columns)

regressor = RandomForestRegressor(max_depth=50,n_estimators=500, random_state =0,criterion='absolute_error',)
regressor.fit(iq_train_x,iq_train_y)
iq_train_pred = regressor.predict(iq_train_x)
iq_val_pred = regressor.predict(iq_val_x)

rmse=(mean_absolute_error(iq_train_y,iq_train_pred))
print("MAE train:",rmse)
rmse=(mean_absolute_error(iq_val_y,iq_val_pred))
print ("MAE validation:",rmse)
examine(iq_val_pred,iq_val_y)

iq_test = rawfeats[rawfeats.city=='iq'].copy()
iq_test = iq_test[features]
iq_test = iq_test.interpolate(method='linear', limit_direction='forward')
iq_norm = scaler.fit_transform(iq_test)
iq_test = pd.DataFrame(iq_norm, columns=iq_test.columns)
iq_test_preds = regressor.predict(iq_test)
iq_test_preds = iq_test_preds.astype(int)

submission = pd.read_csv('data/submission_format.csv',index_col=[0, 1, 2])
submission.total_cases = np.concatenate([sj_test_preds, iq_test_preds])
submission.to_csv("submissions/randomforest.csv")

