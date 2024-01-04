import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
from datetime import datetime
import features
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


sj = features.sj
features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c','week_start_date']

sj_train_x, sj_test_x, sj_train_y, sj_test_y = train_test_split(sj[features], sj['total_cases'], test_size=0.2, random_state=12345)
keep_dates = sj_train_x['week_start_date']
sj_train_x = sj_train_x.drop('week_start_date',axis=1)
sj_test_x = sj_test_x.drop('week_start_date',axis=1)


scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(sj_train_x)
sj_train_x = pd.DataFrame(normalized_data, columns=sj_train_x.columns)
normalized_data = scaler.fit_transform(sj_test_x)
sj_test_x = pd.DataFrame(normalized_data, columns=sj_test_x.columns)
knn_model = KNeighborsRegressor(n_neighbors=10)
knn_model.fit(sj_train_x, sj_train_y)

train_preds = knn_model.predict(sj_train_x)
mse = mean_squared_error(sj_train_y, train_preds)
rmse = sqrt(mse)
print(rmse)
test_preds = knn_model.predict(sj_test_x)
mse = mean_squared_error(sj_test_y, test_preds)
rmse = sqrt(mse)

print(rmse)

train_preds_df = pd.DataFrame(train_preds, columns=['total_cases_predicted'])

sj_train_y = pd.DataFrame(sj_train_y, columns=['total_cases'])


#keep_dates = keep_dates.drop('week_start_date',axis=0)


train_preds_df["week_start_date"] = keep_dates.values
sj_train_x["week_start_date"] = keep_dates.values


       # Plotting predictions versus y_train for sj_data
plt.plot(sj_train_x['week_start_date'], sj_train_y['total_cases'], label='Actual total_cases', color='blue')
plt.plot(train_preds_df['week_start_date'], train_preds_df['total_cases_predicted'], label='Predicted total_cases', color='red')


plt.title('Predictions vs Actual for sj_data')
plt.show()
   
# plt.xlabel('Week Start Date')
#     plt.ylabel('Total Cases')
#     plt.legend()
# else:
#     print("Error: Sizes of test_sj['total_cases'] and y_pred_sj are different.")

# # Calculate and print mean absolute error (MAE) and mean squared error (MSE) for sj_data
# mae_sj = mean_absolute_error(test_sj['total_cases'], y_pred_sj)
# mse_sj = mean_squared_error(test_sj['total_cases'], y_pred_sj)
# print(f'Mean Absolute Error (MAE) for sj_data: {mae_sj}')
# print(f'Mean Squared Error (MSE) for sj_data: {mse_sj}')