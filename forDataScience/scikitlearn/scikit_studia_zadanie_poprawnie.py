# Poniżej umieść swoje rozwiązanie
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

houses = pd.read_csv("houses.csv")

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

Q1 = houses['Cena'].quantile(0.25)
Q3 = houses['Cena'].quantile(0.75)

IQR = Q3-Q1

upper_bound = Q3 + IQR * 1.5
lower_bound = Q1 - IQR * 1.5

houses = houses[(houses['Cena'] >= lower_bound) & (houses['Cena'] <= upper_bound)]

encoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')
encoded = encoder.fit_transform(houses[cat_features])
encoded = pd.DataFrame(encoded)

encoded.index = houses.index
encoded.columns = encoder.get_feature_names_out(cat_features)

houses = houses.drop(columns = cat_features)
houses = houses.join(encoded)

from sklearn.model_selection import train_test_split

X = houses.drop(columns = ["Cena"])
y = houses['Cena']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_features.remove('Cena')

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
}

# Słowniki do przechowywania wyników
metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    metrics["Model"].append(name)
    metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
    metrics["RMSE"].append(sqrt(mean_squared_error(y_test, y_pred)))
    metrics["R² Score"].append(r2_score(y_test, y_pred))

    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_