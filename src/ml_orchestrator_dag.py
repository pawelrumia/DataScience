from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# 1. Definicja domyślnych ustawień dla procesów Airflow
default_args = {
    'owner': 'pawelrumia',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 2. Inicjalizacja głównego drzewa procesów (DAG)
with DAG(
    'melbourne_banking_ml_pipeline',
    default_args=default_args,
    description='Automatyczny potok codziennego retrenowania modeli ML w DataScience',
    schedule_interval=None, # Uruchamianie ręczne lub na żądanie, w produkcji np. '@daily'
    catchup=False,
) as dag:

    # Zadanie 1: Uruchomienie analizy finansowej w Spark SQL
    run_spark_analysis = BashOperator(
        task_id='run_spark_bank_analysis',
        bash_command='python /workspace/src/bank_spark_analysis.py',
    )

    # Zadanie 2: Trening klasyfikatora XGBoost (startuje po analizie)
    train_bank_xgb = BashOperator(
        task_id='train_bank_loan_xgb',
        bash_command='python /workspace/src/bank_loan_xgb.py',
    )

    # Zadanie 3: Trening sieci neuronowej Keras (startuje równolegle z XGBoostem)
    train_keras_nn = BashOperator(
        task_id='train_melbourne_keras_nn',
        bash_command='python /workspace/src/dl_nn_model.py',
    )

    # 3. Ustawienie hierarchii i zależności (Dyrygowanie procesem)
    # Najpierw idzie analiza Spark, a po jej sukcesie równolegle ruszają dwa treningi ML!
    run_spark_analysis >> [train_bank_xgb, train_keras_nn]
