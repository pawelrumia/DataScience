from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import accuracy_score

# 1) Wczytanie danych i szybki rzut oka
df = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\heart.csv")

# 2) Wybór kolumny celu i podział na X/y
X = df.drop("target", axis=1)
y = df["target"]

# 3) Train/test split 80/20 z losowaniem kontrolowanym i stratyfikacją
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Listy kolumn numerycznych i kategorycznych (wg dtype)
numeric_features = X.select_dtypes(include=['int64', "float64"]).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 5) Transformatory dla typów zmiennych
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 6) ColumnTransformer łączący oba pipeline'y
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Funkcja budująca pipeline: preprocessor -> classifier
def build_pipeline(model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),  # Twój ColumnTransformer z poprzedniego kroku
        ("classifier", model)
    ])


# Modele # Pipeline'y
log_reg = LogisticRegression(max_iter=1000)
log_reg_pipeline = build_pipeline(log_reg)

rf = RandomForestClassifier(random_state=42)
rf_pipeline = build_pipeline(rf)

svm = SVC()  # w razie potrzeby probabilistycznych wyjść: SVC(probability=True)
svm_pipeline = build_pipeline(svm)

# Siatka parametrów dla regresji logistycznej
param_grid = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__penalty": ['l2'],
}

# GridSearchCV na pipeline z regresją logistyczną
grid_search_log_reg = GridSearchCV(
    estimator=log_reg_pipeline,
    param_grid=param_grid,
    cv=5
)

# Dopasowanie na zbiorze treningowym
grid_search_log_reg.fit(X_train, y_train)

# Najlepsze parametry
lr_best = grid_search_log_reg.best_params_

# Siatka parametrów dla RandomForest w pipeline (prefix 'classifier__')
param_dist = {
    "classifier__n_estimators": [10, 50, 100],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

# RandomizedSearchCV na pipeline z lasem losowym
random_search_rf = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42
)

# Dopasowanie na danych treningowych
random_search_rf.fit(X_train, y_train)

# Najlepsze parametry
rf_best = random_search_rf.best_params_

# Siatka parametrów dla SVM
param_grid_svm = {
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": [1, 0.1, 0.01],
    "classifier__kernel": ["rbf", "linear"],
}

# GridSearchCV na pipeline z SVM
grid_search_svm = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid_svm,
    cv=5
)

# Dopasowanie na danych treningowych
grid_search_svm.fit(X_train, y_train)

# Najlepsze parametry
svm_best = grid_search_svm.best_params_

# 1) Start klastra H2O
h2o.init()  # opcjonalnie: h2o.init(nthreads=-1, max_mem_size="2G")

# 2) Sklej X i y w jedną ramkę pandas
train_pd = pd.concat([X_train, y_train], axis=1)
test_pd = pd.concat([X_test, y_test], axis=1)

# 3) Konwersja do H2OFrame
train_h2o = h2o.H2OFrame(train_pd)
test_h2o = h2o.H2OFrame(test_pd)

# 4) Klasyfikacja: ustaw cel jako kategoryczny
train_h2o["target"] = train_h2o["target"].asfactor()
test_h2o["target"] = test_h2o["target"].asfactor()

# (opcjonalnie) nazwy predyktorów i celu do późniejszego trenowania w H2O
# response = "target"
# predictors = [c for c in train_h2o.columns if c != response]

# Zmienne pomocnicze
response = "target"
predictors = [c for c in train_h2o.columns if c != response]

# Obiekt AutoML
aml = H2OAutoML(
    seed=42,
    max_runtime_secs=60,
    nfolds=5
)

# Trening
aml.train(y=response, training_frame=train_h2o)

# Tablica rankingowa (leaderboard)
lb = aml.leaderboard  # H2OFrame z modelami posortowanymi po metryce
# (opcjonalnie jako pandas)
# lb_pd = lb.as_data_frame()

# 1) Predykcje scikit-learn (używamy dopasowanych obiektów search)
y_pred_lr = grid_search_log_reg.predict(X_test)
y_pred_rf = random_search_rf.predict(X_test)
y_pred_svm = grid_search_svm.predict(X_test)

# 2) Predykcje H2O AutoML na teście
preds_h2o = aml.leader.predict(test_h2o)  # H2OFrame z kolumnami: predict, p0, p1
y_true_h2o = test_h2o["target"].as_data_frame(use_pandas=True).iloc[:, 0].astype(str).values
y_pred_h2o = preds_h2o["predict"].as_data_frame(use_pandas=True).iloc[:, 0].astype(str).values

# 3) Słownik z accuracy dla każdego modelu
accuracy_scores = {
    "Logistic Regression": accuracy_score(y_test, grid_search_log_reg.best_estimator_.named_steps["classifier"].predict(
        X_test)),
    "Random Forest": accuracy_score(y_test, random_search_rf.best_estimator_.named_steps["classifier"].predict(X_test)),
    "SVM": accuracy_score(y_test, grid_search_svm.best_estimator_.named_steps["classifier"].predict(X_test)),
    "H2O AutoML": accuracy_score(y_test, aml.leader.predict(test_h2o).as_data_frame()["predict"])
}

# 4) Ramka danych Model / Accuracy
accuracy_df = pd.DataFrame(accuracy_scores.items(), columns=["Model", "Accuracy"]
)
