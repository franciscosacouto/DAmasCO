import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
from datetime import datetime
import features
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

iq = features.iq
sj = features.sj
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
features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c','weekofyear']

sj_train_x, sj_val_x, sj_train_y, sj_val_y = train_test_split(sj[features], sj['total_cases'], test_size=0.2, random_state=12345)



scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(sj_train_x)
sj_train_x = pd.DataFrame(normalized_data, columns=sj_train_x.columns)
normalized_data = scaler.fit_transform(sj_val_x)
sj_val_x = pd.DataFrame(normalized_data, columns=sj_val_x.columns)
knn_model = KNeighborsRegressor(n_neighbors=10)
knn_model.fit(sj_train_x, sj_train_y)

train_preds = knn_model.predict(sj_train_x)
mse = mean_squared_error(sj_train_y, train_preds)
rmse = sqrt(mse)
print("SAN JUAN")
print("train data" , rmse)
MAE= mean_absolute_error(sj_train_y,train_preds)
print("MAE train:", MAE)

val_preds = knn_model.predict(sj_val_x)
mse = mean_squared_error(sj_val_y, val_preds)
rmse_test = sqrt(rmse)
print("test data" , rmse_test)
MAE_test= mean_absolute_error(sj_val_y,val_preds)
print("MAE test:", MAE_test)


sj_train_y = pd.DataFrame(sj_train_y, columns=['total_cases'])


examine(val_preds,sj_val_y)






rawfeats = pd.read_csv('data/dengue_features_test.csv')
sj_test = rawfeats[rawfeats.city=='sj'].copy()
sj_test = sj_test[features]
sj_test = sj_test.interpolate(method='linear', limit_direction='forward')
sj_norm = scaler.fit_transform(sj_test)
sj_test = pd.DataFrame(sj_norm, columns=sj_test.columns)
sj_test_preds = knn_model.predict(sj_test)
sj_test_preds = sj_test_preds.astype(int)


features = ['reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','station_avg_temp_c','station_min_temp_c','weekofyear']

iq_train_x, iq_val_x, iq_train_y, iq_val_y = train_test_split(iq[features], iq['total_cases'], test_size=0.2, random_state=12345)




normalized_data = scaler.fit_transform(iq_train_x)
iq_train_x = pd.DataFrame(normalized_data, columns=iq_train_x.columns)
normalized_data = scaler.fit_transform(iq_val_x)
iq_val_x = pd.DataFrame(normalized_data, columns=iq_val_x.columns)
knn_model = KNeighborsRegressor(n_neighbors=10)
knn_model.fit(iq_train_x, iq_train_y)

iq_train_preds = knn_model.predict(iq_train_x)
iq_test_preds= knn_model.predict(iq_val_x)
mse = mean_squared_error(iq_train_y, iq_train_preds)
print("IQUITOS")
rmse = sqrt(mse)
print("train rmse" , rmse)
MAE= mean_absolute_error(iq_train_y,iq_train_preds)
print("MAE train:", MAE)
mse = mean_squared_error(iq_val_y, iq_test_preds)
rmse_test = sqrt(rmse)
print("test data" , rmse_test)
MAE_test= mean_absolute_error(iq_val_y,iq_test_preds)
print("MAE test:", MAE_test)

examine(iq_test_preds,iq_val_y)




iq_test = rawfeats[rawfeats.city=='iq'].copy()
iq_test = iq_test[features]
iq_test = iq_test.interpolate(method='linear', limit_direction='forward')
iq_norm = scaler.fit_transform(iq_test)
iq_test = pd.DataFrame(iq_norm, columns=iq_test.columns)
iq_test_preds = knn_model.predict(iq_test)
iq_test_preds = iq_test_preds.astype(int)

submission = pd.read_csv('data/submission_format.csv',index_col=[0, 1, 2])
submission.total_cases = np.concatenate([sj_test_preds, iq_test_preds])
submission.to_csv("submissions/knn.csv")


