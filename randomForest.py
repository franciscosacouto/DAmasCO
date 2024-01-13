import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

sj=features.sj

features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c',"weekofyear"]

sj_train_x, sj_test_x, sj_train_y, sj_test_y = train_test_split(sj[features], sj['total_cases'], test_size=0.2, random_state=28)
regressor = RandomForestRegressor(n_estimators=200, random_state = 0)
regressor.fit(sj_train_x,sj_train_y)
train_pred = regressor.predict(sj_train_x)
test_pred = regressor.predict(sj_test_x)

rmse=(mean_absolute_error(sj_train_y,train_pred))
print("rmse train:",rmse)
rmse=(mean_absolute_error(sj_test_y,test_pred))
print ("rmse test:",rmse)


