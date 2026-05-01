"""
data_ingestion.py
-----------------
Handles data ingestion from:
  - AWS S3
  - Microsoft Fabric Lakehouse (ADLS Gen2)
  - PostgreSQL
  - MongoDB
  - Local CSV files
"""

import os
import io
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─── S3 Ingestion ─────────────────────────────────────────────────────────────
class S3Ingestion:
    def __init__(self):
        import boto3

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        self.bucket = os.getenv("S3_BUCKET", "lumen-health-insurance")

    def read_csv(self, key: str) -> pd.DataFrame:
        logger.info(f"Reading s3://{self.bucket}/{key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        logger.info(f"Writing {len(df)} rows to s3://{self.bucket}/{key}")
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())

    def list_files(self, prefix: str) -> list:
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def ingest_all(self) -> dict:
        """Ingest all datasets from S3 and return as dict of DataFrames."""
        datasets = {}
        files = {
            "members": "raw/members.csv",
            "claims": "raw/claims.csv",
            "providers": "raw/providers.csv",
            "notes": "raw/clinical_notes.csv",
        }
        for name, key in files.items():
            try:
                datasets[name] = self.read_csv(key)
                logger.info(f"  ✓ {name}: {len(datasets[name]):,} rows")
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e} — falling back to local")
                datasets[name] = pd.read_csv(
                    f"data/synthetic/{name.replace('members','members').replace('notes','clinical_notes')}.csv"
                )
        return datasets


# ─── Microsoft Fabric / ADLS Gen2 Ingestion ───────────────────────────────────
class FabricIngestion:
    """
    Reads/writes data from Microsoft Fabric Lakehouse via ADLS Gen2.
    In a Fabric Notebook, use spark.read directly — this client is for
    external scripts and CI/CD pipelines.
    """

    def __init__(self):
        from azure.storage.blob import BlobServiceClient

        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={os.getenv('ADLS_ACCOUNT_NAME')};"
            f"AccountKey={os.getenv('ADLS_ACCOUNT_KEY')};"
            f"EndpointSuffix=core.windows.net"
        )
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container = os.getenv("ADLS_CONTAINER", "health-data")

    def read_parquet(self, blob_path: str) -> pd.DataFrame:
        logger.info(f"Reading from Fabric Lakehouse: {blob_path}")
        blob = self.client.get_blob_client(container=self.container, blob=blob_path)
        data = blob.download_blob().readall()
        return pd.read_parquet(io.BytesIO(data))

    def write_parquet(self, df: pd.DataFrame, blob_path: str) -> None:
        logger.info(f"Writing {len(df):,} rows → Fabric Lakehouse: {blob_path}")
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        blob = self.client.get_blob_client(container=self.container, blob=blob_path)
        blob.upload_blob(buffer.getvalue(), overwrite=True)
        logger.info(f"  ✓ Upload complete: {blob_path}")

    def get_spark_reader(self, spark_session, path: str):
        """For use inside Fabric Notebook — read via Spark."""
        full_path = f"abfss://{self.container}@{os.getenv('ADLS_ACCOUNT_NAME')}.dfs.core.windows.net/{path}"
        return spark_session.read.parquet(full_path)


# ─── PostgreSQL Ingestion ─────────────────────────────────────────────────────
class PostgreSQLIngestion:
    def __init__(self):
        from sqlalchemy import create_engine

        url = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
            f"/{os.getenv('POSTGRES_DB')}"
        )
        self.engine = create_engine(url)

    def read_table(self, table: str, where: str = "") -> pd.DataFrame:
        query = f"SELECT * FROM {table}"
        if where:
            query += f" WHERE {where}"
        logger.info(f"Querying: {query[:80]}...")
        return pd.read_sql(query, self.engine)

    def read_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)

    def write_table(
        self, df: pd.DataFrame, table: str, if_exists: str = "append"
    ) -> None:
        logger.info(f"Writing {len(df):,} rows → {table}")
        df.to_sql(table, self.engine, if_exists=if_exists, index=False, chunksize=1000)


# ─── MongoDB Ingestion ────────────────────────────────────────────────────────
class MongoIngestion:
    def __init__(self):
        from pymongo import MongoClient

        self.client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.client["health_insurance"]

    def read_collection(
        self, collection: str, query: dict = {}, limit: int = 0
    ) -> pd.DataFrame:
        logger.info(f"Reading MongoDB collection: {collection}")
        cursor = self.db[collection].find(query, {"_id": 0}).limit(limit)
        return pd.DataFrame(list(cursor))

    def insert_many(self, collection: str, records: list) -> None:
        logger.info(f"Inserting {len(records):,} documents → {collection}")
        self.db[collection].insert_many(records)

    def upsert(self, collection: str, key_field: str, record: dict) -> None:
        self.db[collection].update_one(
            {key_field: record[key_field]},
            {"$set": record},
            upsert=True,
        )


# ─── Local CSV Ingestion (dev / fallback) ─────────────────────────────────────
class LocalIngestion:
    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> dict:
        datasets = {}
        file_map = {
            "members": "members.csv",
            "claims": "claims.csv",
            "providers": "providers.csv",
            "notes": "clinical_notes.csv",
        }
        for name, fname in file_map.items():
            path = self.data_dir / fname
            if path.exists():
                datasets[name] = pd.read_csv(path)
                logger.info(f"  ✓ Loaded {name}: {len(datasets[name]):,} rows")
            else:
                logger.warning(f"  ✗ File not found: {path}")
        return datasets


# ─── Unified Ingestion Factory ────────────────────────────────────────────────
class DataIngestionFactory:
    """
    Auto-selects ingestion source based on environment config.
    Priority: Fabric → S3 → PostgreSQL → Local
    """

    @staticmethod
    def get_data(source: str = "auto") -> dict:
        if source == "auto":
            if os.getenv("ADLS_ACCOUNT_NAME"):
                source = "fabric"
            elif os.getenv("AWS_ACCESS_KEY_ID"):
                source = "s3"
            elif os.getenv("POSTGRES_HOST"):
                source = "postgres"
            else:
                source = "local"

        logger.info(f"📥 Using ingestion source: {source.upper()}")

        if source == "fabric":
            ingestion = FabricIngestion()
            return {
                "members": ingestion.read_parquet("processed/members.parquet"),
                "claims": ingestion.read_parquet("processed/claims.parquet"),
                "providers": ingestion.read_parquet("processed/providers.parquet"),
                "notes": ingestion.read_parquet("processed/clinical_notes.parquet"),
            }
        elif source == "s3":
            return S3Ingestion().ingest_all()
        elif source == "postgres":
            pg = PostgreSQLIngestion()
            return {
                "members": pg.read_table("members"),
                "claims": pg.read_table("claims"),
                "providers": pg.read_table("providers"),
                "notes": pg.read_table("clinical_notes"),
            }
        else:
            return LocalIngestion().load_all()


if __name__ == "__main__":
    data = DataIngestionFactory.get_data(source="local")
    for name, df in data.items():
        print(f"{name}: {df.shape} | columns: {list(df.columns)[:5]}")
