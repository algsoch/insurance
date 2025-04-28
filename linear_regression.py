# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2. Load dataset
df = pd.read_csv('insurance.csv')  # adjust path if needed

# 3. Explore dataset
print(df.head())
print(df.describe())
print(df.info())

# 4. Check for missing values
print(df.isnull().sum())

# 5. Visualize correlations
sns.pairplot(df, hue='smoker')
plt.show()

# 6. Define Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# 7. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Build Preprocessing Pipeline
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')  # drop to avoid dummy variable trap

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 9. Build Linear Regression Model Pipeline
linreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])

# 10. Train Model
linreg_pipeline.fit(X_train, y_train)

# 11. Predict
y_pred = linreg_pipeline.predict(X_test)

# 12. Evaluate
print('Linear Regression Results:')
print('R2 Score:', r2_score(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# 13. Ridge Regression
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('ridge', Ridge())])

params_ridge = {'ridge__alpha': [0.1, 1.0, 10.0, 50.0]}
ridge_cv = GridSearchCV(ridge_pipeline, params_ridge, cv=5)
ridge_cv.fit(X_train, y_train)

print('Best Ridge Alpha:', ridge_cv.best_params_)

y_pred_ridge = ridge_cv.predict(X_test)
print('Ridge R2 Score:', r2_score(y_test, y_pred_ridge))
print('Ridge RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# 14. Lasso Regression
lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('lasso', Lasso(max_iter=5000))])

params_lasso = {'lasso__alpha': [0.001, 0.01, 0.1, 1.0]}
lasso_cv = GridSearchCV(lasso_pipeline, params_lasso, cv=5)
lasso_cv.fit(X_train, y_train)

print('Best Lasso Alpha:', lasso_cv.best_params_)

y_pred_lasso = lasso_cv.predict(X_test)
print('Lasso R2 Score:', r2_score(y_test, y_pred_lasso))
print('Lasso RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

# 15. Save final model for Streamlit
import joblib
joblib.dump(linreg_pipeline, 'insurance_model.pkl')
