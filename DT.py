import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Change to DecisionTreeRegressor for regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
import features

sj=features.sj
iq=features.iq

# Specify the columns you want to include in the new DataFrame
selected_columns = [
    'station_max_temp_c',  
    'reanalysis_dew_point_temp_k',
    'reanalysis_relative_humidity_percent',
    'reanalysis_specific_humidity_g_per_kg',
    'weekofyear',
    'total_cases'
]

# Create a new DataFrame with the selected columns
df = sj[selected_columns]

X_train, X_test, y_train, y_test = train_test_split(df.drop('total_cases', axis=1), df['total_cases'], random_state=0)

clf = DecisionTreeRegressor(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Use mean_absolute_error for regression
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Plot the decision tree
fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(clf, 
                   feature_names=list(df.columns[:-1]),  # Exclude the target variable
                   filled=True)
plt.show()