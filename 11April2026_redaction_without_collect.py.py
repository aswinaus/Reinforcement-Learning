# Databricks notebook source
from pyspark.sql import functions as F, types as T

# -----------------------------------------
# 1. Downstream reads upstream parquet files
# -----------------------------------------
input_path = "abfss://stage@account.dfs.core.windows.net/extracted_text_parquet/"

df = spark.read.parquet(input_path)

# Expected shape:
# document_id | page_number | chunk_number | text | image_paths | metadata...

# -----------------------------------------
# 2. Keep only what is needed for redaction
# -----------------------------------------
text_df = df.select(
    "document_id",
    "page_number",
    "chunk_number",
    "text"
).filter(F.col("text").isNotNull())

# -----------------------------------------
# 3. Repartition for distributed processing
# -----------------------------------------
text_df = text_df.repartition(64)

# -----------------------------------------
# 4. Redact on executors using mapPartitions
#    so Presidio objects are initialized once
#    per partition, not once per row
# -----------------------------------------
output_schema = T.StructType([
    T.StructField("document_id", T.StringType(), True),
    T.StructField("page_number", T.IntegerType(), True),
    T.StructField("chunk_number", T.IntegerType(), True),
    T.StructField("original_text", T.StringType(), True),
    T.StructField("redacted_text", T.StringType(), True),
    T.StructField("pii_found", T.BooleanType(), True)
])

def redact_partition(rows):
    # Runs inside each executor partition
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    for row in rows:
        text = row["text"] or ""

        findings = analyzer.analyze(
            text=text,
            language="en"
        )

        redacted = anonymizer.anonymize(
            text=text,
            analyzer_results=findings
        ).text

        yield (
            row["document_id"],
            row["page_number"],
            row["chunk_number"],
            text,
            redacted,
            len(findings) > 0
        )

redacted_rdd = text_df.rdd.mapPartitions(redact_partition)
redacted_df = spark.createDataFrame(redacted_rdd, schema=output_schema)

# -----------------------------------------
# 5. Write distributed output back to storage
# -----------------------------------------
output_path = "abfss://redacted@account.dfs.core.windows.net/pii_redacted_parquet/"

redacted_df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(output_path)