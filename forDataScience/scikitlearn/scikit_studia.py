
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.python.ops.losses.losses_impl import mean_squared_error

houses = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\houses.csv")
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.to_list()
cat_features = houses.select_dtypes(include=['object']).columns.to_list()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

correlation_matrix = houses[num_features].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Macierz korelacji cech numerycznych")
plt.tight_layout()
plt.show()

Q1 = houses['Cena'].quantile(0.25)
Q3 = houses['Cena'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

houses[(houses['Cena'] <= lower_bound) | (houses['Cena'] >= upper_bound)].shape[0]
houses = houses[(houses['Cena'] >= lower_bound) & (houses['Cena'] <= upper_bound)]

plt.figure(figsize=(10, 6))
sns.boxplot(x=houses['Cena'], palette='coolwarm')
plt.title('Cena')
plt.savefig("cena_bez_outlinerow")

plt.figure(figsize=(8, 6))
sns.histplot(houses['Cena'], bins=30, kde=True, color='blue')
plt.title('Histogram')
plt.xlabel('Cena')
plt.ylabel('Ilosc domow')
plt.savefig("histogram_bez_outlinerow")

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index

encoded.columns = encoder.get_feature_names_out(cat_features)

houses = houses.drop(columns=cat_features).join(encoded)

y = houses['Cena']
X = houses.drop(columns=['Cena'])

# 2) Podział 80/20 z random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()
num_features.remove("Cena")
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
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
    metrics["RMSE"].append(mean_squared_error(y_test, y_pred))
    metrics["R² Score"].append(r2_score(y_test, y_pred))

    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_


