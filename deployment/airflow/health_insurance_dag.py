"""
health_insurance_dag.py
-----------------------
Apache Airflow DAG for the Health Insurance AI Platform.
Runs daily: ingest → preprocess → train → evaluate → deploy
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.email import send_email

# ─── Default Args ─────────────────────────────────────────────────────────────
default_args = {
    "owner":            "lumen-data-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email":            ["ml-alerts@lumentechnologies.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    "health_insurance_ml_pipeline",
    default_args=default_args,
    description="Daily ML pipeline: ingest → train → deploy",
    schedule_interval="0 2 * * *",   # 2 AM daily
    catchup=False,
    tags=["health-insurance", "ml", "lumen"],
)


# ─── Task Functions ───────────────────────────────────────────────────────────

def ingest_data(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/health-insurance-ai")
    from src.ingestion.data_ingestion import DataIngestionFactory
    data = DataIngestionFactory.get_data(source="auto")
    context["ti"].xcom_push(key="row_counts",
                             value={k: len(v) for k, v in data.items()})
    print(f"Ingested: {[(k, len(v)) for k, v in data.items()]}")


def preprocess_data(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/health-insurance-ai")
    from src.ingestion.data_ingestion import DataIngestionFactory
    from src.preprocessing.feature_engineering import PreprocessingPipeline
    data = DataIngestionFactory.get_data(source="auto")
    pipeline = PreprocessingPipeline()
    processed = pipeline.fit_transform_all(data)
    context["ti"].xcom_push(key="feature_shapes", value={
        "claim_features": str(processed["claim_features"].shape),
        "fraud_features": str(processed["fraud_features"].shape),
    })


def train_claim_model(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/health-insurance-ai")
    from src.ingestion.data_ingestion import DataIngestionFactory
    from src.preprocessing.feature_engineering import PreprocessingPipeline
    from src.models.claim_approval_model import ClaimApprovalModel
    data = DataIngestionFactory.get_data(source="auto")
    processed = PreprocessingPipeline().fit_transform_all(data)
    model = ClaimApprovalModel(model_type="xgboost")
    metrics = model.train(processed["claim_features"], processed["claim_labels"])
    context["ti"].xcom_push(key="claim_auc", value=metrics["auc_roc"])
    print(f"Claim model AUC: {metrics['auc_roc']}")


def train_fraud_model(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/health-insurance-ai")
    from src.ingestion.data_ingestion import DataIngestionFactory
    from src.preprocessing.feature_engineering import PreprocessingPipeline
    from src.models.fraud_detection_model import FraudDetectionModel
    data = DataIngestionFactory.get_data(source="auto")
    processed = PreprocessingPipeline().fit_transform_all(data)
    model = FraudDetectionModel()
    model.train_isolation_forest(processed["fraud_features"])
    model.train_kmeans(processed["fraud_features"], processed["fraud_labels"])
    model.train_supervised(processed["fraud_features"], processed["fraud_labels"])
    model.save()
    print("Fraud model trained and saved.")


def evaluate_models(**context):
    """Check model AUC thresholds — fail DAG if below minimum."""
    claim_auc = context["ti"].xcom_pull(key="claim_auc", task_ids="train_claim_model")
    MIN_AUC = 0.85
    if claim_auc and float(claim_auc) < MIN_AUC:
        raise ValueError(f"Claim model AUC {claim_auc} below threshold {MIN_AUC}!")
    print(f"✅ Model evaluation passed | Claim AUC: {claim_auc}")


def deploy_api(**context):
    """Restart FastAPI service after new model artifacts are saved."""
    import subprocess
    result = subprocess.run(
        ["systemctl", "restart", "health-insurance-api"],
        capture_output=True, text=True
    )
    print(f"API restart: {result.returncode} | {result.stdout}")


# ─── Tasks ────────────────────────────────────────────────────────────────────

t_ingest = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_data,
    dag=dag,
)

t_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

t_train_claim = PythonOperator(
    task_id="train_claim_model",
    python_callable=train_claim_model,
    dag=dag,
)

t_train_fraud = PythonOperator(
    task_id="train_fraud_model",
    python_callable=train_fraud_model,
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate_models",
    python_callable=evaluate_models,
    dag=dag,
)

t_deploy = PythonOperator(
    task_id="deploy_api",
    python_callable=deploy_api,
    dag=dag,
)

t_notify = BashOperator(
    task_id="notify_team",
    bash_command=(
        'echo "Pipeline complete: $(date)" | '
        'mail -s "Health Insurance ML Pipeline Done" ml-team@lumentechnologies.com || true'
    ),
    dag=dag,
)

# ─── Dependencies ─────────────────────────────────────────────────────────────
t_ingest >> t_preprocess >> [t_train_claim, t_train_fraud] >> t_evaluate >> t_deploy >> t_notify
