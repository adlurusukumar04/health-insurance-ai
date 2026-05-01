"""
fabric_pipeline.py
-------------------
Microsoft Fabric / PySpark pipeline.
Run this inside a Microsoft Fabric Notebook for large-scale processing.

Steps:
  1. Read raw data from Fabric Lakehouse (ADLS Gen2)
  2. Preprocess with Spark
  3. Feature engineering
  4. Train with Azure ML
  5. Write predictions back to Lakehouse
"""

# ── In Fabric Notebooks, SparkSession is pre-created as `spark` ──────────────
# If running locally, create your own:
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("HealthInsuranceAI").getOrCreate()

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline as SparkPipeline
import os

# ─── Config ───────────────────────────────────────────────────────────────────
LAKEHOUSE_PATH = "abfss://health-data@lumenadls.dfs.core.windows.net"
RAW_PATH       = f"{LAKEHOUSE_PATH}/raw"
PROCESSED_PATH = f"{LAKEHOUSE_PATH}/processed"
MODELS_PATH    = f"{LAKEHOUSE_PATH}/models"


# ─── Step 1: Read from Lakehouse ──────────────────────────────────────────────
def read_data():
    print("📥 Reading data from Fabric Lakehouse...")
    claims    = spark.read.parquet(f"{RAW_PATH}/claims/")
    members   = spark.read.parquet(f"{RAW_PATH}/members/")
    providers = spark.read.parquet(f"{RAW_PATH}/providers/")

    print(f"  Claims:    {claims.count():,} rows")
    print(f"  Members:   {members.count():,} rows")
    print(f"  Providers: {providers.count():,} rows")
    return claims, members, providers


# ─── Step 2: Spark Preprocessing ─────────────────────────────────────────────
def preprocess(claims, members, providers):
    print("⚙️  Preprocessing with Spark...")

    # Join claims with members
    df = claims.join(
        members.select("member_id", "age", "bmi", "smoker",
                        "plan_type", "num_chronic_conditions"),
        on="member_id", how="left"
    )

    # Feature engineering
    df = df.withColumn("claim_approved",
            F.when(F.col("claim_status") == "Approved", 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("high_cost_flag",
            F.when(F.col("claim_amount") > 15000, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("composite_risk",
            (F.col("age") / 85 * 0.3 +
             F.col("num_chronic_conditions") / 3 * 0.4 +
             F.col("smoker").cast(DoubleType()) * 0.1 +
             F.col("high_cost_flag") * 0.2).cast(DoubleType()))
    df = df.fillna({"bmi": 25.0, "num_chronic_conditions": 0})

    return df


# ─── Step 3: Spark ML Pipeline (Claim Approval) ───────────────────────────────
def train_claim_model(df):
    print("🤖 Training Claim Approval model (Spark GBT)...")

    # Index categorical columns
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in ["claim_type", "plan_type", "state", "diagnosis_code"]
    ]

    feature_cols = [
        "claim_amount", "age", "bmi", "num_chronic_conditions",
        "num_procedures", "days_in_hospital", "prior_auth",
        "high_cost_flag", "composite_risk",
        "claim_type_idx", "plan_type_idx", "state_idx", "diagnosis_code_idx"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                                 handleInvalid="skip")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features")
    gbt       = GBTClassifier(featuresCol="features", labelCol="claim_approved",
                               maxIter=100, maxDepth=5, seed=42)

    pipeline = SparkPipeline(stages=indexers + [assembler, scaler, gbt])

    # Train / test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="claim_approved", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    print(f"  ✅ GBT Claim Model AUC: {auc:.4f}")

    # Save model to Lakehouse
    model.save(f"{MODELS_PATH}/claim_approval_gbt")
    print(f"  Model saved → {MODELS_PATH}/claim_approval_gbt")

    return model, auc


# ─── Step 4: Write Predictions Back ──────────────────────────────────────────
def write_predictions(model, df):
    print("📤 Writing predictions to Lakehouse...")
    preds = model.transform(df).select(
        "claim_id", "member_id",
        F.col("probability").getItem(1).alias("approval_probability").cast(DoubleType()),
        F.col("prediction").alias("predicted_approved").cast(IntegerType()),
        F.current_timestamp().alias("predicted_at"),
    )
    (preds.write
          .mode("overwrite")
          .parquet(f"{PROCESSED_PATH}/claim_predictions/"))
    print(f"  ✅ {preds.count():,} predictions written")
    return preds


# ─── Spark SQL Analytics ──────────────────────────────────────────────────────
def run_analytics(df):
    print("📊 Running analytics queries...")
    df.createOrReplaceTempView("claims")

    # Monthly claim trends
    trend = spark.sql("""
        SELECT
            DATE_TRUNC('month', claim_date)   AS month,
            claim_type,
            COUNT(*)                          AS num_claims,
            SUM(claim_amount)                 AS total_amount,
            AVG(claim_amount)                 AS avg_amount,
            SUM(CAST(is_fraud AS INT))        AS fraud_count
        FROM claims
        GROUP BY 1, 2
        ORDER BY 1 DESC
    """)
    trend.show(10)
    trend.write.mode("overwrite").parquet(f"{PROCESSED_PATH}/analytics/monthly_trends/")

    # Provider fraud risk
    provider_risk = spark.sql("""
        SELECT
            provider_id,
            COUNT(*)                                              AS total_claims,
            AVG(claim_amount)                                     AS avg_claim,
            SUM(CAST(is_fraud AS INT))                            AS fraud_claims,
            ROUND(SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
        FROM claims
        GROUP BY provider_id
        ORDER BY fraud_rate_pct DESC
        LIMIT 50
    """)
    provider_risk.show(10)
    return trend, provider_risk


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n🏥 Health Insurance AI – Microsoft Fabric Pipeline\n" + "="*50)
    try:
        claims, members, providers = read_data()
        df = preprocess(claims, members, providers)
        model, auc = train_claim_model(df)
        write_predictions(model, df)
        run_analytics(df)
        print(f"\n✅ Fabric pipeline complete | Claim Model AUC: {auc:.4f}")
    except NameError:
        print("⚠️  SparkSession not found. Run this inside a Microsoft Fabric Notebook.")
        print("    Or initialize: spark = SparkSession.builder.appName('test').getOrCreate()")


if __name__ == "__main__":
    main()
