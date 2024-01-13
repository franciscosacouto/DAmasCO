import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from boruta import BorutaPy
import features

sj = features.sj
iq = features.iq
dengue_labels_train = features.rawlabels
dengue_data = sj
dengue_features_test = pd.read_csv('data/dengue_features_test.csv')

# Feature Selection
boruta_features = dengue_data.copy()
boruta_features['total_cases'] = dengue_labels_train['total_cases']

# Drop 'week_start_date' during Boruta feature selection
X_boruta = boruta_features.drop(columns=['total_cases', 'week_start_date'])
y_boruta = boruta_features['total_cases']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# One-hot encoding for 'city'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['city'])], remainder='passthrough')
X_boruta_encoded = ct.fit_transform(X_boruta)

# Boruta Feature Selection
boruta_selector = BorutaPy(DecisionTreeRegressor(), n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(X_boruta_encoded, y_boruta.values)

# Get the selected features
selected_features_mask = boruta_selector.support_
selected_features = ct.get_feature_names_out(X_boruta.columns[selected_features_mask])

pruning = None

# Now, when building the Decision Tree model, handle 'week_start_date' separately
for city in ['sj', 'iq']:
    # Split data
    normalized_dengue_city = dengue_data[dengue_data['city'] == city].copy()
    normalized_test_city = dengue_features_test[dengue_features_test['city'] == city].copy()

    # Handle 'week_start_date' separately
    X_city = normalized_dengue_city[selected_features].drop(columns=['city', 'year', 'week_start_date'])
    X_city['week_start_date'] = pd.to_datetime(normalized_dengue_city['week_start_date'])  # Convert to datetime
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