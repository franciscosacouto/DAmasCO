import pandas as pd
from statsmodels.regression import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Loading the packages
# Note: In Python, you usually import libraries when you use them, so we don't need to load all at once.
# We will import them as needed.

# Importing data into Python environment
train = pd.read_csv('data/dengue_features_train.csv')
train_results = pd.read_csv('data/dengue_labels_train.csv')
complete_train = train.copy()
complete_train['total_cases'] = train_results.iloc[:, 3]
test = pd.read_csv('data/dengue_features_test.csv')

# Replacing NAs
columns_to_interpolate = [
    'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
    'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_diur_temp_rng_c',
    'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm', 'reanalysis_tdtr_k'
]

for col in columns_to_interpolate:
    complete_train[col] = complete_train[col].interpolate(method='spline', order=3)

# Correlation among columns
numeric_columns = complete_train.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = complete_train[numeric_columns].corr()

# Separate the data by cities
sj_data1 = complete_train[complete_train['city'] == 'sj'].copy()
iq_data = complete_train[complete_train['city'] == 'iq'].copy()

# Removing columns
sj_data = sj_data1.drop(columns=['week_start_date', 'city', 'year'])
iq_data = iq_data.drop(columns=['week_start_date', 'city', 'year'])

sj_data2 = sj_data1.drop(columns=['city', 'year'])
# Splitting the training data in 80:20
train_sj, test_sj = train_test_split(sj_data, test_size=0.2, random_state=100)

# Fitting a model for train data
X_train_sj = train_sj.drop(columns=['total_cases'])
y_train_sj = train_sj['total_cases']
fit = linear_model.OLS(y_train_sj, X_train_sj).fit()
# print(fit.summary())

# Retrieve variable names
variable_names = X_train_sj.columns
coefficients = fit.params

# Display variable names and their coefficients
for var, coef in zip(variable_names, coefficients):
    print(f"{var}: {coef}")


# Predicting the model
X_test_sj = test_sj.drop(columns=['total_cases'])
y_pred_sj = fit.predict(X_test_sj)
# print(y_pred_sj.corr(test_sj['total_cases']))





# Calculate and print mean absolute error (MAE) and mean squared error (MSE) for sj_data
mae_sj = mean_absolute_error(test_sj['total_cases'], y_pred_sj)
mse_sj = mean_squared_error(test_sj['total_cases'], y_pred_sj)
print(f'Mean Absolute Error (MAE) for sj_data: {mae_sj}')
print(f'Mean Squared Error (MSE) for sj_data: {mse_sj}')

# Replacing NAs in test data
for col in columns_to_interpolate:
    not_null_indices = test[col].notnull()
    x = not_null_indices.index[not_null_indices].to_numpy()
    y = test.loc[not_null_indices, col].to_numpy()
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    test[col] = f(test.index)

# Dividing original test data for different cities
sj_test_data = test[test['city'] == 'sj'].copy()
iq_test_data = test[test['city'] == 'iq'].copy()
sj_test_data = sj_test_data.drop(columns=['week_start_date', 'city', 'year'])
iq_test_data = iq_test_data.drop(columns=['week_start_date', 'city', 'year'])

# Fitting and predicting complete model for entire train data
X_sj = sj_data.drop(columns=['total_cases'])
y_sj = sj_data['total_cases']
fit_sj = linear_model.OLS(y_sj, X_sj).fit()
# print(fit_sj.summary())
y_pred_sj = fit_sj.predict(sj_test_data)
y_pred_sj.to_csv("sj.csv", index=False)

X_iq = iq_data.drop(columns=['total_cases'])
y_iq = iq_data['total_cases']
fit_iq = linear_model.OLS(y_iq, X_iq).fit()
# print(fit_iq.summary())
y_pred_iq = fit_iq.predict(iq_test_data)
y_pred_iq.to_csv("iq.csv", index=False)

# Step AIC final submission
autofit_sj = fit_sj
aic_sj = autofit_sj.aic
y_pred_sj = autofit_sj.predict(sj_test_data)
y_pred_sj.to_csv("sj.csv", index=False)

autofit_iq = fit_iq
aic_iq = autofit_iq.aic
y_pred_iq = autofit_iq.predict(iq_test_data)
y_pred_iq.to_csv("iq.csv", index=False)






# Predicting the model
X_test_sj = sj_data.drop(columns=['total_cases'])
y_pred_sj = fit.predict(X_test_sj)

sj_data2['week_start_date'] = pd.to_datetime(sj_data2['week_start_date'])
# Ensure the sizes of test_sj['total_cases'] and y_pred_sj are the same
if len(sj_data['total_cases']) == len(y_pred_sj):
       # Plotting predictions versus y_train for sj_data
    plt.plot(sj_data2['week_start_date'], y_pred_sj, label='Predicted total_cases', color='red')
    plt.plot(sj_data2['week_start_date'], sj_data['total_cases'], label='Actual total_cases', color='blue')
    plt.title('Predictions vs Actual for sj_data')
    plt.xlabel('Week Start Date')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.show()
else:
    print("Error: Sizes of test_sj['total_cases'] and y_pred_sj are different.")



