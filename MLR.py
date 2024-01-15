#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import features

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
sj = features.sj
iq= features.iq

#---sj---

#Setting the value for X and Y
attributes = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c']
x = sj[attributes]
y = sj['total_cases']

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
mlr.fit(x_train, y_train)

#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)

#Graphs
examine(y_pred_mlr,y_test)

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff.head())

#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

#prediction model
rawfeats = pd.read_csv('data/dengue_features_test.csv')
sj_validation = rawfeats[rawfeats.city=='sj'].copy()
sj_validation = sj_validation[attributes]
sj_validation = sj_validation.interpolate(method='linear', limit_direction='forward')
y_validation_mlr_sj= mlr.predict(sj_validation)
y_validation_mlr_sj = y_validation_mlr_sj.astype(int)


#---iq---

#Setting the value for X and Y
attributes = ['reanalysis_dew_point_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',  'station_min_temp_c']
x = iq[attributes]
y = iq['total_cases']

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
mlr.fit(x_train, y_train)

#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)

#Graphs
examine(y_pred_mlr,y_test)

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff.head())

#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

#Prediction model
rawfeats = pd.read_csv('data/dengue_features_test.csv')
iq_validation = rawfeats[rawfeats.city=='iq'].copy()
iq_validation = iq_validation[attributes]
iq_validation = iq_validation.interpolate(method='linear', limit_direction='forward')
y_validation_mlr_iq= mlr.predict(iq_validation)
y_validation_mlr_iq = y_validation_mlr_iq.astype(int)

#Submission
submission = pd.read_csv('data/submission_format.csv',index_col=[0, 1, 2])
submission.total_cases = np.concatenate([y_validation_mlr_sj, y_validation_mlr_iq])
submission.to_csv("submissions/mlr.csv")