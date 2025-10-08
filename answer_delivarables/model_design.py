import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

file_path = os.path.join('data', 'listings_sample.csv')
df = pd.read_csv(file_path)

columns_to_drop = ['id', 'address', 'media_url']
df = df.drop(columns=columns_to_drop, errors='ignore')
df = df.dropna()

X = df.drop(columns=['price'], errors='ignore')
y = df['price']

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, None],
    'model__max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()

print("\nTraining Set Evaluation:")
print(f"  MSE: {train_mse:.2f}")
print(f"  R2: {train_r2:.2f}")

print("\nTesting Set Evaluation:")
print(f"  MSE: {test_mse:.2f}")
print(f"  R2: {test_r2:.2f}")

print("\nCross-Validation Evaluation:")
print(f"  CV MSE: {cv_mse:.2f}")

if abs(train_r2 - test_r2) > 0.1:
    print("\nWarning: The model may be overfitting.")
elif train_r2 < 0.5 or test_r2 < 0.5:
    print("\nWarning: The model may be underfitting.")
else:
    print("\nThe model appears to be well-fitted.")