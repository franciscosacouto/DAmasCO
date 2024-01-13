import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree


# Import Data
dengue_data = pd.read_csv('data/dengue_features_train.csv')
dengue_labels_train = pd.read_csv('data/dengue_labels_train.csv')
dengue_features_test = pd.read_csv('data/dengue_features_test.csv')
submission_format = pd.read_csv('data/submission_format.csv')

# Data Cleaning
na_count = dengue_data.isna().sum()
unnecessary_feature = na_count[na_count > 25].index
dengue_data = dengue_data.drop(columns=unnecessary_feature)
dengue_features_test = dengue_features_test.drop(columns=unnecessary_feature)

for col in dengue_data.columns[4:]:
    dengue_data[col] = dengue_data[col].interpolate()

# Mean Values of Data
#aggregated = dengue_data.groupby('city').mean().T

numeric_columns = dengue_data.select_dtypes(include=['number']).columns
aggregated = dengue_data[numeric_columns].groupby('city').mean().T
aggregated.columns = ['iq', 'sj']

# Correlation Matrix
num_dengue_data = dengue_data.drop(columns=['city', 'week_start_date'])
correlation_matrix = num_dengue_data.corr()

high_correlated_feature = correlation_matrix[correlation_matrix > 0.90].stack().index
high_correlated_feature = set(col for col in high_correlated_feature if col[0] != col[1])

dengue_data = dengue_data.drop(columns=high_correlated_feature)
dengue_features_test = dengue_features_test.drop(columns=high_correlated_feature)

# Data Normalization
for col in dengue_data.columns[4:]:
    if 'temp_k' in col:
        dengue_data[col] = dengue_data[col] - 273.15

for col in dengue_features_test.columns[4:]:
    if 'temp_k' in col:
        dengue_features_test[col] = dengue_features_test[col] - 273.15

    temp_mean = dengue_data[col].mean()
    temp_std = dengue_data[col].std()

    dengue_data[col] = (dengue_data[col] - temp_mean) / temp_std

# Data Pre-processing for Test Dataset (Submission)
for col in dengue_features_test.columns[4:]:
    dengue_features_test[col] = dengue_features_test[col].interpolate()

    if 'temp_k' in col:
        dengue_features_test[col] = dengue_features_test[col] - 273.15

    temp_mean = dengue_features_test[col].mean()
    temp_std = dengue_features_test[col].std()

    dengue_features_test[col] = (dengue_features_test[col] - temp_mean) / temp_std

dengue_features_test['weekofyear'] = dengue_features_test['weekofyear'].astype('category')

# Feature Selection
boruta_features = dengue_data.copy()
boruta_features['total_cases'] = dengue_labels_train['total_cases']

X_boruta = boruta_features.drop(columns=['total_cases'])
y_boruta = boruta_features['total_cases']

# Boruta Feature Selection
from boruta import BorutaPy

boruta_selector = BorutaPy(DecisionTreeRegressor(), n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(X_boruta.values, y_boruta.values)

selected_features = X_boruta.columns[boruta_selector.support_].tolist()

# Prediction Model with Decision Tree
result_df = pd.DataFrame(columns=['Model for SJ', 'Model for IQ', 'MAE'])

for model_type in ['Unpruned Tree', 'Pruned Tree']:
    if model_type == 'Unpruned Tree':
        pruning = None
    else:
        pruning = 'best'

    for city in ['sj', 'iq']:
        # Split data
        normalized_dengue_city = dengue_data[dengue_data['city'] == city].copy()
        normalized_test_city = dengue_features_test[dengue_features_test['city'] == city].copy()

        X_city = normalized_dengue_city[selected_features].drop(columns=['city', 'year', 'week_start_date'])
        y_city = normalized_dengue_city['total_cases']

        X_train_city, X_test_city, y_train_city, y_test_city = train_test_split(X_city, y_city, test_size=0.2, shuffle=False)

        # Build Decision Tree Model
        tree_model = DecisionTreeRegressor(criterion='mae', ccp_alpha=pruning)
        tree_model.fit(X_train_city, y_train_city)

        # Model Evaluation
        y_pred_city = tree_model.predict(X_test_city)
        mae_city = mean_absolute_error(y_test_city, y_pred_city)

        result_df = result_df.append({'Model for SJ': model_type if city == 'sj' else None,
                                      'Model for IQ': model_type if city == 'iq' else None,
                                      'MAE': mae_city}, ignore_index=True)

        # Plot Decision Tree
        plt.figure(figsize=(15, 10))
        tree.plot_tree(tree_model, feature_names=X_train_city.columns, filled=True, rounded=True)
        plt.show()

# Predict the submission data
submission_pred_df = pd.DataFrame(columns=['Model Type', 'Prediction'])
for model_type, pruning in zip(['Pruned Tree for SJ, Unpruned Tree for IQ', 'Pruned Tree for Both Cities', 'Unpruned Tree for Both Cities'],
                               ['best', 'best', None]):
    pred_sj = tree_model.predict(dengue_features_test[dengue_features_test['city'] == 'sj'][selected_features].drop(columns=['city', 'year', 'week_start_date']))
    pred_iq = tree_model.predict(dengue_features_test[dengue_features_test['city'] == 'iq'][selected_features].drop(columns=['city', 'year', 'week_start_date']))
    submission_pred = pd.concat([pd.Series(pred_sj), pd.Series(pred_iq)]).reset_index(drop=True)
    submission_pred_df = submission_pred_df.append({'Model Type': model_type, 'Prediction': submission_pred}, ignore_index=True)

# Result
result_df.to_csv("result.csv", index=False)
submission_pred_df.to_csv("submission_pred.csv", index=False)