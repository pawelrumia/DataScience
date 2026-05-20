import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

import joblib


# =========================
# LOAD DATA
# =========================

df = pd.read_csv('data/Melbourne_housing.csv')

# =========================
# CONVERT NUMERIC COLUMNS
# =========================

numeric_columns = [
    'Rooms',
    'Price',
    'Distance',
    'Week',
    'CPI',
    'LastCPI',
    'Distance2'
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# =========================
# DATE
# =========================

df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['Date'])

df = df.dropna()

print(df.head())

print(df.info())


# =========================
# CLEAN DATA
# =========================

df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    errors='coerce'
)

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month

df['price_per_room'] = df['Price'] / df['Rooms']

df = df.drop(columns=['Date'])


# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(10, 6))

sns.histplot(df['Price'], bins=50)

plt.title('Price Distribution')

plt.savefig('images/price_distribution.png')

plt.close()


# =========================
# REMOVE OUTLIERS
# =========================

q1 = df['Price'].quantile(0.25)
q3 = df['Price'].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[
    (df['Price'] >= lower) &
    (df['Price'] <= upper)
]


# =========================
# FEATURES / TARGET
# =========================

X = df.drop(columns=['Price'])

y = df['Price']


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# COLUMN TYPES
# =========================

numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_features = X.select_dtypes(
    include=['object']
).columns


# =========================
# PREPROCESSING
# =========================

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# =========================
# MODEL
# =========================

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =========================
# TRAIN
# =========================

model.fit(X_train, y_train)


# =========================
# PREDICT
# =========================

predictions = model.predict(X_test)


# =========================
# METRICS
# =========================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print('\\nRESULTS:')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')


# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    'models/melbourne_model.pkl'
)

print('\\nModel saved!')

# =========================
# FEATURE IMPORTANCE
# =========================

regressor = model.named_steps['regressor']

importances = regressor.feature_importances_

feature_names = (
    numeric_features.tolist()
)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances[:len(feature_names)]
})

importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

print('\\nFEATURE IMPORTANCE:')
print(importance_df)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df,
    x='importance',
    y='feature'
)

plt.title('Feature Importance')

plt.savefig('images/feature_importance.png')

plt.close()