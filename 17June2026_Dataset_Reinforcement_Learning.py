# Databricks notebook source
# MAGIC %md
# MAGIC # Dataset for Model Learning v2.0 (Reinforcement Fine-Tuning)
# MAGIC
# MAGIC Prepares **training** and **validation** datasets for Azure OpenAI **Reinforcement Fine-Tuning (RFT)** of `o4-mini` using a **multi-grader architecture** (RLAIF pattern).
# MAGIC
# MAGIC ## Architecture: RLAIF with Human Reference Data
# MAGIC
# MAGIC **Pattern:** Human reviewer (ground truth) → gpt-4o (judge) → o4-mini (student)
# MAGIC
# MAGIC - o4-mini generates its own outputs during training
# MAGIC - A **structural Python grader** validates JSON format and required fields
# MAGIC - A **semantic gpt-4o grader** scores quality against human-corrected reference
# MAGIC - Final reward: `0.15 × structural + 0.85 × semantic`
# MAGIC
# MAGIC ## Data Sources
# MAGIC
# MAGIC - `hive_metastore.default.kh_asset` — Delta table with human-corrected fields and raw LLM output
# MAGIC - `redacted_adls_path` column → redacted source document text in ADLS (parquet)
# MAGIC - `Asset_Creation_Tax_V7.4.10` — System prompt used during asset creation
# MAGIC - `cleanup_asset_json` notebook — `extract_analysis_results()` pattern for parsing `asset_json`
# MAGIC
# MAGIC ## RFT Training Format (JSONL)
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "messages": [{"role": "user", "content": "<system prompt>\n\n<source document>"}],
# MAGIC   "Asset": "<human-corrected, cleaned>",
# MAGIC   "Client_issue": "<human-corrected, cleaned>",
# MAGIC   "AB_IP_generated": "<human-corrected, cleaned>",
# MAGIC   "Tax_years": "2024",
# MAGIC   "References": ["Art 10(3)", "WHT Act 2021"],
# MAGIC   "original_model_Asset": "<extracted from asset_json via extract_analysis_results()>",
# MAGIC   "original_model_Client_issue": "<extracted from asset_json>",
# MAGIC   "original_model_AB_IP_generated": "<extracted from asset_json>",
# MAGIC   "review_comment": "<human reviewer feedback>",
# MAGIC   "approval_status": "approved|rejected|unreviewed"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ## Column Mapping
# MAGIC
# MAGIC | Source | Field | Purpose |
# MAGIC | --- | --- | --- |
# MAGIC | kh_asset columns | `Asset`, `Client_Issue`, `AB_Analysis`, `Tax_Years`, `References` | Human-corrected reference (ground truth) |
# MAGIC | kh_asset `asset_json` | Extracted via `extract_analysis_results()` → `Asset`, `Client_issue`, `AB_IP_generated` | Original model output (before correction) |
# MAGIC | kh_asset `redacted_adls_path` | Parquet path to source document | Input text for model |
# MAGIC | kh_asset `is_approved` | `True` / `False` | Reviewer approval signal |
# MAGIC | kh_asset `review_comment` | Free text | Reviewer feedback for semantic grader |
# MAGIC
# MAGIC ## Data Cleaning (from `cleanup_asset_json`)
# MAGIC
# MAGIC - `extract_analysis_results()` — Unwraps `{ok, text: "<json>"}` envelope, strips markdown fences, extracts `analysis_results[0]`
# MAGIC - `clean_value()` — Removes bullet-point formatting (`\n•`, `\n-`, `\n*`), collapses whitespace, recursively cleans lists/dicts
# MAGIC
# MAGIC ## What Happens During Training (RFT Loop)
# MAGIC
# MAGIC During reinforcement fine-tuning, Azure OpenAI runs a repeated loop for each training example across multiple epochs:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │  FOR EACH TRAINING EXAMPLE:                                         │
# MAGIC │                                                                     │
# MAGIC │  1. INPUT: o4-mini receives the source document + system prompt     │
# MAGIC │     (from the "messages" field in the JSONL)                        │
# MAGIC │                                                                     │
# MAGIC │  2. GENERATE: o4-mini produces a NEW tax asset analysis             │
# MAGIC │     (JSON with Asset, Client_issue, AB_IP_generated, etc.)          │
# MAGIC │     This output is different every time — the model is exploring.   │
# MAGIC │                                                                     │
# MAGIC │  3. STRUCTURAL GRADER (Python, runs in Azure sandbox):              │
# MAGIC │     Receives: o4-mini's new output                                  │
# MAGIC │     Checks: Is it valid JSON? Are the 5 KH fields present?          │
# MAGIC │     Returns: 0.0–1.0 format score                                   │
# MAGIC │                                                                     │
# MAGIC │  4. SEMANTIC GRADER (gpt-4o, runs as score_model):                  │
# MAGIC │     Receives:                                                       │
# MAGIC │       • Human-corrected reference (our ground truth columns)        │
# MAGIC │       • Original model output (what o4-mini generated BEFORE        │
# MAGIC │         human correction — extracted from asset_json)                │
# MAGIC │       • Reviewer comments + approval status                         │
# MAGIC │       • o4-mini's NEW output (from step 2)                          │
# MAGIC │     Evaluates: How close is the new output to human ground truth?   │
# MAGIC │     Returns: 0.0–1.0 quality score                                  │
# MAGIC │                                                                     │
# MAGIC │  5. REWARD = 0.15 × structural + 0.85 × semantic                    │
# MAGIC │                                                                     │
# MAGIC │  6. UPDATE: Azure uses this reward to adjust o4-mini's weights      │
# MAGIC │     High reward → reinforce this behavior (do more of this)         │
# MAGIC │     Low reward → discourage this behavior (do less of this)         │
# MAGIC └─────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Key clarification:** gpt-4o is NOT generating a competing response. It is purely the **judge**. It reads o4-mini's new output and scores it against the human ground truth we provide in the training data. The three-way comparison in the semantic grader prompt (human reference vs original model output vs new model output) gives gpt-4o the full context to assess whether the student model is improving, regressing, or hallucinating.
# MAGIC
# MAGIC **Why include the original model output?** It shows gpt-4o what o4-mini got WRONG before human correction. If the new output repeats the same mistakes, gpt-4o can penalise appropriately. If the new output fixes those mistakes (aligning with the human correction), it gets rewarded.
# MAGIC
# MAGIC **Why can't we just use the existing `asset_json` output instead of generating new ones?**
# MAGIC
# MAGIC The `asset_json` column already contains what o4-mini originally generated for each source document. So why not just score that directly instead of making the model generate again? Because that would be **Supervised Fine-Tuning (SFT)**, not Reinforcement Fine-Tuning:
# MAGIC
# MAGIC | Approach | How it works | Limitation |
# MAGIC | --- | --- | --- |
# MAGIC | **SFT** (show correct answer) | "Here's the input, here's what you should output" — model learns to copy | Memorises specific answers. Doesn't learn WHY an answer is good. Brittle on unseen documents. |
# MAGIC | **DPO** (our v1 approach) | "Here's a good output and a bad output — prefer the good one" | Better than SFT, but still learns from fixed examples. Limited exploration. |
# MAGIC | **RFT** (current approach) | Model generates its OWN output, gets scored, adjusts weights based on reward | Learns the underlying principles. Explores the space of possible outputs. Generalises to new documents. |
# MAGIC
# MAGIC During RFT, the model's weights change after every update. So the output it produces in epoch 2 is different from epoch 1, which is different from the original `asset_json`. The training loop needs to evaluate the model's **current** behavior (what it generates NOW with updated weights), not its historical behavior. This exploration is what teaches o4-mini to reason toward good outputs rather than just memorising 10 correct answers.
# MAGIC
# MAGIC The `asset_json` original output is still used — but as context for the **judge** (gpt-4o), not as the thing being scored. It tells the judge: "here's what the model used to get wrong, check if it's still making the same mistakes."
# MAGIC
# MAGIC ## Why Multi-Grader Architecture?
# MAGIC
# MAGIC We evolved through three iterations to arrive at this design. The initial approach used a single Python grader with string-comparison scoring (exact match and substring containment between model output and human reference). This failed at scale with **23% runtime errors** (`exit_code=202`, `FileNotFoundError`) because the grader could not reliably parse the variety of output formats the model produces during training — raw JSON, `{ok, text: "..."}` wrappers, markdown-fenced blocks, and preamble text before the JSON. A hardened parser alone was insufficient because string comparison is semantically blind: two outputs can express the same tax analysis in different words and score 0.0. Conversely, a model can copy surface tokens while hallucinating facts and score 1.0. The multi-grader separates these concerns: the **structural grader** answers "is this valid, well-formed output?" (cheap, deterministic, never crashes), while the **semantic grader** answers "is this correct, complete, and aligned with the human reviewer?" (expensive but semantically aware). This also mirrors the knowledge hierarchy: deterministic validation gates the reward signal so that gpt-4o only judges outputs that are at least structurally sound.
# MAGIC
# MAGIC ## Multi-Grader Architecture
# MAGIC
# MAGIC ### Structural Grader (Python, weight 0.15)
# MAGIC - No file I/O, no external dependencies
# MAGIC - 3-fallback JSON parser: direct parse → regex extraction → brace-balancing
# MAGIC - Handles: raw JSON, `{ok, text}` wrapper, `analysis_results[]` wrapper, markdown fences
# MAGIC - Scoring: JSON parseable (0.30) + fields present (0.30 × N/5) + references valid (0.20) + tax year valid (0.20)
# MAGIC - Signal encoding: `0.0` = unparseable, `0.01` = no KH fields, `0.02–0.99` = partial, `1.0` = perfect
# MAGIC
# MAGIC ### Semantic Grader (gpt-4o score_model, weight 0.85)
# MAGIC - Evaluates model output against human-corrected reference
# MAGIC - Receives: human reference, original model output (from `asset_json`), reviewer comments + approval status
# MAGIC - Returns a single score 0.0–1.0
# MAGIC
# MAGIC **Reward Calculation — Evaluation Dimensions:**
# MAGIC
# MAGIC The semantic grader converts reviewer feedback and reference comparison into five structured evaluation dimensions, each with a defined weight:
# MAGIC
# MAGIC | Reviewer Feedback Pattern | Evaluation Dimension | Weight |
# MAGIC | --- | --- | --- |
# MAGIC | "wrong classification", Asset/Client_issue mismatch | `classification_accuracy` | 35% |
# MAGIC | "hallucinated content", facts not in source reference | `factuality` | 25% |
# MAGIC | "missed edge cases", missing references, incomplete analysis | `completeness` | 20% |
# MAGIC | "too verbose", unclear language, poor tax-domain framing | `professional_quality` | 10% |
# MAGIC | Reviewer rejected with specific comments not addressed | `reviewer_alignment` | 10% |
# MAGIC
# MAGIC **Penalty Deductions (hard caps):**
# MAGIC
# MAGIC | Penalty Trigger | Score Deduction |
# MAGIC | --- | --- |
# MAGIC | Wrong classification (model's Asset name / Client_issue does not match human reference) | −0.35 |
# MAGIC | Hallucinated facts not present in the human-corrected AB_IP_generated | −0.30 |
# MAGIC | Missing references that appear in the human version | −0.25 |
# MAGIC | Ignoring reviewer feedback (reviewer rejected and gave specific comments that are unaddressed) | −0.20 |
# MAGIC
# MAGIC **How it works:** gpt-4o receives the human-corrected reference fields, the original model output (extracted from `asset_json` via `extract_analysis_results()`), reviewer comments, and the new model output to score. It evaluates how close the new output is to the human ground truth across all five dimensions, applies penalties for critical failures, and returns a single decimal score. This score is weighted at 0.85 in the final reward formula.
# MAGIC
# MAGIC ## Pre-Training Grader Validation
# MAGIC
# MAGIC Before submitting the fine-tuning job, we validate the structural grader locally by running it against a suite of synthetic test cases that represent the full spectrum of model outputs — from perfect to completely broken. This ensures the grader behaves predictably and will not crash or produce misleading scores during training (which would corrupt the reward signal).
# MAGIC
# MAGIC The test suite covers:
# MAGIC
# MAGIC | Test Case | Input | Expected Score | What it validates |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Perfect output | Valid JSON with all 5 KH fields, valid tax year, references list | 1.00 | Grader correctly rewards well-formed output |
# MAGIC | Unparseable garbage | Plain text, no JSON structure | 0.00 | Grader returns zero (not crash) for broken output |
# MAGIC | JSON but no KH fields | Valid JSON like `{"foo": "bar"}` with none of the required fields | 0.01 | Distinguishes "valid JSON, wrong content" from unparseable |
# MAGIC | Wrapped in `{ok, text}` | Double-encoded JSON inside the pipeline wrapper | 0.90–1.00 | `extract_analysis_results()` pattern correctly unwraps nested envelopes |
# MAGIC | Missing refs and tax_years | Valid JSON with only 3 of 5 fields, no references or tax year | 0.30–0.70 | Partial credit scales correctly with field coverage |
# MAGIC | Markdown fenced | JSON wrapped in ` ```json ... ``` ` fences | 0.90–1.00 | Fence stripping works before parsing |
# MAGIC
# MAGIC If any test case scores outside its expected range, the cell raises an `AssertionError` and blocks submission — preventing a broken grader from being sent to Azure OpenAI where it would cause the 20% runtime error threshold to be exceeded (as happened in v1).
# MAGIC
# MAGIC This validation step is what prevents the `exit_code=202` / `FileNotFoundError` failures we saw in earlier iterations: the grader is proven crash-proof against every known output format before it ever runs at scale during training.
# MAGIC
# MAGIC ## Output Location
# MAGIC
# MAGIC ADLS `stage` container → `fine-tuning/dpo/<timestamp>/train.jsonl` + `validation.jsonl`
# MAGIC
# MAGIC ## Authentication
# MAGIC
# MAGIC OAuth2 via Service Principal (`keyvault-secret-scope`)

# COMMAND ----------

# ============================================================
# ADLS Authentication & Configuration
# Same Service Principal credentials as Azure Resource Cost notebook
# Reference: /Users/aswin@eyaswin.onmicrosoft.com/Azure Resource Cost and Compute Utilization
# ============================================================

tenant_id = dbutils.secrets.get(scope="keyvault-secret-scope", key="Oauth-Tenant-ID")
client_id = dbutils.secrets.get(scope="keyvault-secret-scope", key="Oauth-Client-ID")
client_secret = dbutils.secrets.get(scope="keyvault-secret-scope", key="Oauth-Client-Secret")
storage_account_name = dbutils.secrets.get(scope="keyvault-secret-scope", key="Storage-Account-Name")

# Configure Spark for ADLS Gen2 access
domain = f"{storage_account_name}.dfs.core.windows.net"
oauth_endpoint = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"

spark.conf.set(f"fs.azure.account.auth.type.{domain}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{domain}", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{domain}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{domain}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{domain}", oauth_endpoint)

# Helper for ADLS paths
ABFSS = lambda container, path="": f"abfss://{container}@{storage_account_name}.dfs.core.windows.net/{path}"

print(f"\u2705 ADLS authenticated: {storage_account_name}")
print(f"   Tenant: {tenant_id[:8]}...")
print(f"   Client: {client_id[:8]}...")

# COMMAND ----------

# ============================================================
# Load kh_asset Delta Table
# Contains: generated assets (RAG pipeline output) and source text
# ============================================================

from pyspark.sql import functions as F

# Read the kh_asset table
df_kh = spark.table("hive_metastore.default.kh_asset")

print(f"Total records in kh_asset: {df_kh.count():,}")
print(f"\nSchema:")
df_kh.printSchema()

print(f"\nSample columns (first 5000 rows):")
display(df_kh.limit(5000))

# COMMAND ----------

# # ============================================================
# # Compose redacted_adls_path for ALL kh_asset rows
# # Join to the appropriate file queue based on file extension:
# #   - .pdf  → pdf_file_queue       → cleaned/text/pdf/
# #   - .docx → word_file_queue      → cleaned/text/word/
# #   - .pptx → ppt_file_queue       → cleaned/text/ppt/
# #   - .msg  → email_file_queue     → cleaned/text/email/
# #   - .xlsx → excel_csv_file_queue → cleaned/text/excel/
# # Path transformation:
# #   - Replace 'raw' container with 'redacted'
# #   - Replace 'incoming/' with 'cleaned/text/<type>/'
# # DO NOT update kh_asset yet — display for review first
# # ============================================================
# from pyspark.sql import functions as F
# from functools import reduce

# # Load all file queue tables
# df_pdf_q = spark.table("hive_metastore.default.pdf_file_queue")
# df_word_q = spark.table("hive_metastore.default.word_file_queue")
# df_ppt_q = spark.table("hive_metastore.default.ppt_file_queue")
# df_email_q = spark.table("hive_metastore.default.email_file_queue")
# df_excel_q = spark.table("hive_metastore.default.excel_csv_file_queue")

# # Add filename column to each queue
# def add_filename(df):
#     return df.withColumn("filename", F.element_at(F.split(F.col("path"), "/"), -1))

# df_pdf_q = add_filename(df_pdf_q).select("path", "filename")
# df_word_q = add_filename(df_word_q).select("path", "filename")
# df_ppt_q = add_filename(df_ppt_q).select("path", "filename")
# df_email_q = add_filename(df_email_q).select("path", "filename")
# df_excel_q = add_filename(df_excel_q).select("path", "filename")

# # kh_asset base: asset_id + document_name
# df_kh_base = df_kh.select("asset_id", "document_name")

# # Join each queue and compose redacted_adls_path
# def join_and_compose(df_kh_base, df_queue, ext_filter, cleaned_segment):
#     """Join kh_asset to a file queue and compose the redacted path."""
#     joined = df_kh_base.filter(
#         F.lower(F.col("document_name")).endswith(ext_filter)
#     ).join(
#         df_queue,
#         df_kh_base["document_name"] == df_queue["filename"],
#         "inner"
#     ).drop("filename")
    
#     # Compose redacted path and remove the file extension from the end
#     composed = joined.withColumn(
#         "_redacted_with_ext",
#         F.regexp_replace(
#             F.regexp_replace(F.col("path"), "^abfss://raw@", "abfss://redacted@"),
#             "/incoming/", f"/cleaned/text/{cleaned_segment}/"
#         )
#     ).withColumn(
#         "redacted_adls_path",
#         F.regexp_replace(F.col("_redacted_with_ext"), r"\.[^./]+$", "")
#     ).drop("_redacted_with_ext")
    
#     return composed.select("asset_id", "document_name", "path", "redacted_adls_path")

# # Process each file type
# df_pdf_result = join_and_compose(df_kh_base, df_pdf_q, ".pdf", "pdf")
# df_word_result = join_and_compose(df_kh_base, df_word_q, ".docx", "word")
# df_ppt_result = join_and_compose(df_kh_base, df_ppt_q, ".pptx", "ppt")
# df_email_result = join_and_compose(df_kh_base, df_email_q, ".msg", "email")
# df_excel_result = join_and_compose(df_kh_base, df_excel_q, ".xlsx", "excel")

# # Union all results
# df_all_results = reduce(lambda a, b: a.unionByName(b), [
#     df_pdf_result, df_word_result, df_ppt_result, df_email_result, df_excel_result
# ])

# # Summary
# total_matched = df_all_results.count()
# total_kh = df_kh.count()
# print(f"Total kh_asset rows: {total_kh}")
# print(f"Total matched to file queues: {total_matched}")
# print(f"Unmatched: {total_kh - total_matched}")
# print(f"\nBreakdown by file type:")
# print(f"  .pdf:  {df_pdf_result.count()}")
# print(f"  .docx: {df_word_result.count()}")
# print(f"  .pptx: {df_ppt_result.count()}")
# print(f"  .msg:  {df_email_result.count()}")
# print(f"  .xlsx: {df_excel_result.count()}")

# # Display all results
# print(f"\n{'='*80}")
# print(f"ALL COMPOSED redacted_adls_path VALUES (for review before updating kh_asset):")
# print(f"{'='*80}")
# display(
#     df_all_results.select("document_name", "redacted_adls_path")
#     .orderBy("document_name")
# )

# COMMAND ----------

# # ============================================================
# # Find kh_asset rows that did NOT match any file queue
# # and show their is_approved status
# # ============================================================

# # Get all matched asset_ids (deduplicated)
# matched_asset_ids = df_all_results.select("asset_id").distinct()

# # Left anti-join to find unmatched
# df_unmatched = df_kh.join(matched_asset_ids, on="asset_id", how="left_anti")

# unmatched_count = df_unmatched.count()
# print(f"Unmatched kh_asset rows (no file queue match): {unmatched_count}")
# print(f"\nThese are the documents that did NOT match any file queue:")
# print(f"{'='*80}")

# display(
#     df_unmatched.select(
#         "document_name", "is_approved", "classification_type", "review_comment"
#     ).orderBy("document_name")
# )

# # Also show the count by approval status for the MATCHED rows
# print(f"\n{'='*80}")
# print(f"MATCHED rows approval status:")
# matched_kh = df_kh.join(matched_asset_ids, on="asset_id", how="inner")
# print(f"  Total matched (distinct asset_ids): {matched_kh.count()}")
# print(f"  is_approved=True:  {matched_kh.filter(F.col('is_approved') == True).count()}")
# print(f"  is_approved=False: {matched_kh.filter(F.col('is_approved') == False).count()}")
# print(f"  is_approved=NULL:  {matched_kh.filter(F.col('is_approved').isNull()).count()}")

# COMMAND ----------

# # ============================================================
# # Validate: Can we retrieve content from composed redacted_adls_path?
# # Only test the 11 REVIEWED assets (is_approved = True or False)
# # NOTE: redacted_adls_path needs filename extension REMOVED
# #       because blob storage paths don't include the extension
# # ============================================================
# import os

# # Get reviewed rows with their composed redacted paths
# df_reviewed = df_kh.filter(F.col("is_approved").isNotNull())
# reviewed_ids = [row["asset_id"] for row in df_reviewed.select("asset_id").collect()]

# # Get composed paths for reviewed assets only
# df_reviewed_paths = df_all_results.filter(F.col("asset_id").isin(reviewed_ids))

# # Deduplicate — take first match per asset_id
# from pyspark.sql.window import Window
# w = Window.partitionBy("asset_id").orderBy("path")
# df_reviewed_paths_dedup = df_reviewed_paths.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")

# # Collect paths — join with reviewed to get is_approved (avoid ambiguous cols)
# reviewed_rows = df_reviewed_paths_dedup.select(
#     F.col("asset_id"), F.col("document_name"), F.col("redacted_adls_path")
# ).join(
#     df_reviewed.select(F.col("asset_id").alias("aid"), F.col("is_approved")),
#     F.col("asset_id") == F.col("aid"),
#     "inner"
# ).drop("aid").collect()

# print(f"Testing content retrieval for {len(reviewed_rows)} reviewed assets")
# print(f"{'='*80}")

# results = []
# for row in reviewed_rows:
#     doc_name = row["document_name"]
#     status = "APPROVED" if row["is_approved"] else "REJECTED"
#     composed_path = row["redacted_adls_path"]
    
#     # Remove the filename extension from the path
#     # e.g., .../cleaned/text/pdf/.../document.pdf → .../cleaned/text/pdf/.../document
#     path_no_ext = os.path.splitext(composed_path)[0] if composed_path else None
    
#     # Try to read the parquet directory
#     # The path_no_ext is a folder containing parquet part files
#     content = None
#     error_msg = None
#     try:
#         df_content = spark.read.parquet(path_no_ext)
#         first_row = df_content.limit(1).collect()
#         if first_row:
#             # Get all text content from the first row (concatenate all string columns)
#             row_dict = first_row[0].asDict()
#             content = str(row_dict)[:2000]
#         else:
#             error_msg = "Parquet is empty (0 rows)"
#     except Exception as e:
#         error_msg = str(e)[:150]
    
#     readable = "✅" if content else "❌"
#     char_count = len(content) if content else 0
    
#     results.append({
#         "document_name": doc_name,
#         "status": status,
#         "readable": readable,
#         "chars": char_count,
#         "path_no_ext": path_no_ext,
#         "error": error_msg
#     })
    
#     print(f"\n{readable} [{status:8s}] {doc_name}")
#     print(f"   Path: ...{path_no_ext[-80:] if path_no_ext else 'None'}")
#     if content:
#         print(f"   Content ({char_count} chars): {content[:150]}...")
#     else:
#         print(f"   Error: {error_msg}")

# # Summary
# success_count = sum(1 for r in results if r["readable"] == "✅")
# print(f"\n{'='*80}")
# print(f"SUMMARY: {success_count} / {len(results)} reviewed assets have readable content")

# COMMAND ----------

# # ============================================================
# # Update kh_asset Delta table: write back the composed
# # redacted_adls_path for the 11 REVIEWED assets only
# # ============================================================
# from delta.tables import DeltaTable

# # Get the 11 reviewed asset_ids
# df_reviewed = df_kh.filter(F.col("is_approved").isNotNull())
# reviewed_ids = [row["asset_id"] for row in df_reviewed.select("asset_id").collect()]

# # Get composed paths for reviewed assets, deduplicated
# from pyspark.sql.window import Window
# w = Window.partitionBy("asset_id").orderBy("path")
# df_updates = (
#     df_all_results
#     .filter(F.col("asset_id").isin(reviewed_ids))
#     .withColumn("rn", F.row_number().over(w))
#     .filter(F.col("rn") == 1)
#     .drop("rn")
#     .select("asset_id", "redacted_adls_path")
# )

# print(f"Updating {df_updates.count()} rows in kh_asset with composed redacted_adls_path...")
# print(f"\nPreview of updates:")
# display(df_updates)

# # Perform the MERGE into kh_asset Delta table
# kh_asset_path = f"abfss://delta-tables@{storage_account_name}.dfs.core.windows.net/kh_asset"
# dt_kh = DeltaTable.forPath(spark, kh_asset_path)

# dt_kh.alias("target").merge(
#     df_updates.alias("source"),
#     "target.asset_id = source.asset_id"
# ).whenMatchedUpdate(
#     set={"redacted_adls_path": "source.redacted_adls_path"}
# ).execute()

# print(f"\n\u2705 Successfully updated redacted_adls_path for 11 reviewed assets in kh_asset")

# # Verify
# df_verify = spark.table("hive_metastore.default.kh_asset").filter(
#     F.col("asset_id").isin(reviewed_ids)
# ).select("document_name", "is_approved", "redacted_adls_path")

# print(f"\nVerification:")
# display(df_verify)

# COMMAND ----------

# # ============================================================
# # Compare two HITL-reviewed assets:
# #   - REJECTED: human rejected the o4-mini output
# #   - APPROVED: human approved/corrected the o4-mini output
# # Shows the difference between PREFERRED_COLS and asset_json
# # ============================================================
# import json

# rejected_id = "KH_20260327_1527656653_479e19a0e50f30d6a557bc92c12822f9dc62616e387a41e5b3612bc5b66c360d"
# approved_id = "KH_20260327_1527656653_6ae7b2675cf10f0172fd6798d6094da64abd9779adc64807af19ff5cc56eebb3"

# rows = df_kh.filter(
#     F.col("asset_id").isin([rejected_id, approved_id])
# ).select(
#     "asset_id", "document_name", "is_approved", "review_comment",
#     "Asset", "Client_Issue", "AB_Analysis", "Tax_Years", "References",
#     "asset_json"
# ).collect()

# print(f"Found {len(rows)} rows\n")

# for row in rows:
#     status = "APPROVED ✅" if row["is_approved"] else "REJECTED ❌"
#     print(f"{'='*80}")
#     print(f"{status} | Doc: {row['document_name']}")
#     print(f"Asset ID: ...{row['asset_id'][-20:]}")
#     print(f"\nReview Comment: {row['review_comment']}")
#     print(f"\n{'─'*30} PREFERRED COLS (Human-corrected) {'─'*30}")
#     print(f"Asset:        {str(row['Asset'] or '')[:500]}")
#     print(f"\nClient_Issue: {str(row['Client_Issue'] or '')[:500]}")
#     print(f"\nAB_Analysis:  {str(row['AB_Analysis'] or '')[:500]}")
#     print(f"\nTax_Years:    {row['Tax_Years']}")
#     print(f"\nReferences:   {str(row['References'] or '')[:500]}")
#     print(f"\n{'─'*30} NON-PREFERRED (Raw o4-mini asset_json) {'─'*30}")
#     aj = str(row['asset_json'] or '')
#     # Pretty print if valid JSON
#     try:
#         aj_parsed = json.loads(aj)
#         print(json.dumps(aj_parsed, indent=2, ensure_ascii=False)[:1000])
#     except Exception:
#         print(aj[:1000])
#     print()

# COMMAND ----------

# # ============================================================
# # Check if kh_asset.document_name matches the filename in pdf_file_queue.path
# # ============================================================
# from pyspark.sql import functions as F

# # Extract filename from pdf_file_queue.path
# df_pdf_with_name = df_pdf.withColumn(
#     "filename", F.element_at(F.split(F.col("path"), "/"), -1)
# )

# # Sample kh_asset document_names
# kh_doc_names = [row["document_name"] for row in df_kh.select("document_name").distinct().limit(10).collect()]
# print("Sample kh_asset.document_name values:")
# for name in kh_doc_names:
#     print(f"  {name}")

# # Sample pdf_file_queue filenames (extracted from path)
# pdf_filenames = [row["filename"] for row in df_pdf_with_name.select("filename").distinct().limit(10).collect()]
# print(f"\nSample pdf_file_queue filename (from path):")
# for name in pdf_filenames:
#     print(f"  {name}")

# # Try join on document_name = filename
# join_result = df_kh.join(
#     df_pdf_with_name.select("path", "filename", "redacted_adls_path").alias("pdf"),
#     df_kh["document_name"] == df_pdf_with_name["filename"],
#     "inner"
# )
# print(f"\n{'='*60}")
# print(f"Join on kh_asset.document_name = pdf_file_queue.filename(path):")
# print(f"  Matches: {join_result.count()}")

# # Also try join on redacted_adls_path directly
# join_rdp = df_kh.alias("kh").join(
#     df_pdf.alias("pdf"),
#     F.col("kh.redacted_adls_path") == F.col("pdf.redacted_adls_path"),
#     "inner"
# )
# print(f"\nJoin on kh_asset.redacted_adls_path = pdf_file_queue.redacted_adls_path:")
# print(f"  Matches: {join_rdp.count()}")

# # How many kh_asset rows have non-null redacted_adls_path?
# has_rdp = df_kh.filter(F.col("redacted_adls_path").isNotNull() & (F.col("redacted_adls_path") != "")).count()
# print(f"\nkh_asset rows with non-null redacted_adls_path: {has_rdp} / {df_kh.count()}")

# COMMAND ----------

# # ============================================================
# # Inspect DPO-relevant columns in kh_asset
# # Check how many rows have valid source docs + preferred/non-preferred
# # ============================================================

# # String columns — check non-null and non-empty
# string_cols = ["Asset", "Client_Issue", "AB_Analysis", "Tax_Years", "References", "asset_json", "redacted_adls_path", "review_comment"]
# # Boolean columns — just check non-null
# bool_cols = ["is_approved"]

# print("DPO Column Availability:")
# print("=" * 60)
# for col_name in string_cols:
#     if col_name in df_kh.columns:
#         non_null = df_kh.filter(F.col(col_name).isNotNull() & (F.col(col_name) != "")).count()
#         print(f"  \u2713 {col_name:25s} \u2192 {non_null:,} non-empty rows")
#     else:
#         print(f"  \u2717 {col_name:25s} \u2192 COLUMN NOT FOUND")

# for col_name in bool_cols:
#     if col_name in df_kh.columns:
#         non_null = df_kh.filter(F.col(col_name).isNotNull()).count()
#         true_count = df_kh.filter(F.col(col_name) == True).count()
#         false_count = df_kh.filter(F.col(col_name) == False).count()
#         print(f"  \u2713 {col_name:25s} \u2192 {non_null:,} non-null (True={true_count}, False={false_count})")
#     else:
#         print(f"  \u2717 {col_name:25s} \u2192 COLUMN NOT FOUND")

# # Key metric: how many rows have ALL required DPO fields?
# total = df_kh.count()
# has_preferred = df_kh.filter(
#     F.col("Asset").isNotNull() & (F.col("Asset") != "")
# ).count()
# has_non_preferred = df_kh.filter(
#     F.col("asset_json").isNotNull() & (F.col("asset_json") != "")
# ).count()
# has_source_path = df_kh.filter(
#     F.col("redacted_adls_path").isNotNull() & (F.col("redacted_adls_path") != "")
# ).count()
# has_reviewed = df_kh.filter(F.col("is_approved").isNotNull()).count()

# # Full DPO-eligible: reviewed + has preferred + has non-preferred + has source path
# dpo_eligible = df_kh.filter(
#     F.col("is_approved").isNotNull() &
#     F.col("Asset").isNotNull() & (F.col("Asset") != "") &
#     F.col("asset_json").isNotNull() & (F.col("asset_json") != "") &
#     F.col("redacted_adls_path").isNotNull() & (F.col("redacted_adls_path") != "")
# ).count()

# print(f"\n{'='*60}")
# print(f"DPO ELIGIBILITY SUMMARY (Total rows: {total:,})")
# print(f"{'='*60}")
# print(f"  Has preferred cols (Asset):     {has_preferred:,} / {total}")
# print(f"  Has non-preferred (asset_json): {has_non_preferred:,} / {total}")
# print(f"  Has source path (ADLS):         {has_source_path:,} / {total}")
# print(f"  Has been reviewed:              {has_reviewed:,} / {total}")
# print(f"  \u2192 FULLY DPO-ELIGIBLE:           {dpo_eligible:,} / {total}")
# print(f"\n  \u26a0\ufe0f  Note: Source doc must also be readable from ADLS and >= 500 chars")

# COMMAND ----------

# ============================================================
# Load the Generator System Prompt from Asset_Creation_Tax_V7.4.10
# This is the prompt used during asset creation — becomes the
# "system" message in our fine-tuning dataset.
# ============================================================

import json

nb_path = "/Workspace/Prod/classification/Jobs/Asset_Creation_Tax_V7.4.10.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb_json = json.load(f)

# Extract code cells
cells_raw = [''.join(cell.get('source', [])) for cell in nb_json.get('cells', []) if cell.get('cell_type') == 'code']

# Find and exec the Prompt cell
GENERATOR_SYSTEM_PROMPT = None
for cell_content in cells_raw:
    if "GENERATOR_SYSTEM_PROMPT" in cell_content and "REVIEWER_SYSTEM_PROMPT" in cell_content and "PHASE 1" in cell_content:
        exec(cell_content.strip(), globals())
        print(f"\u2705 Loaded GENERATOR_SYSTEM_PROMPT from Asset_Creation_Tax_V7.4.10")
        print(f"   Prompt length: {len(GENERATOR_SYSTEM_PROMPT):,} chars")
        break

if GENERATOR_SYSTEM_PROMPT is None:
    raise ValueError("Could not find GENERATOR_SYSTEM_PROMPT in the notebook")

# ── Append strict JSON format instruction ──────────────────────────────────────
# The original prompt causes o4-mini to emit {"text": "<json-string>"} instead
# of a bare JSON object.  This suffix enforces the exact structure the grader
# expects so that sample.output_json.analysis_results[0].Asset resolves cleanly.
GENERATOR_SYSTEM_PROMPT += (
    "\n\nCRITICAL OUTPUT FORMAT: "
    "Your ENTIRE response must be a single raw JSON object — no markdown fences, "
    "no prose, no explanation before or after. "
    "The top-level key must be 'analysis_results' containing a list with exactly one object:\n"
    '{"analysis_results": [{"Asset": "<value>", "Client_Issue": "<value>", '
    '"AB_Analysis": "<value>", "Tax_Years": "<value>", "References": ["<ref1>", "<ref2>"]}]}'
)
print(f"\u2705 Format suffix appended — prompt length now: {len(GENERATOR_SYSTEM_PROMPT):,} chars")

print(f"\n   First 200 chars:")
print(f"   {GENERATOR_SYSTEM_PROMPT[:200]}...")

# COMMAND ----------

# ============================================================
# Build Synthetic Training Dataset
#
# Format: Azure OpenAI Fine-Tuning JSONL
# Each row = one training example:
#   system: Generator prompt (from V7.4.10)
#   user: Source document text (input)
#   assistant: Generated asset JSON (ground truth output)
#
# Filters:
#   - Only rows where asset was successfully generated
#   - Only Tax classification (not Controversy)
#   - Exclude very short documents (< 500 chars)
# ============================================================

import pandas as pd

# Reload kh_asset (includes updated redacted_adls_path from Cell 7)
df_kh = spark.table("hive_metastore.default.kh_asset")

# Collect kh_asset data to pandas for processing
df_pd = df_kh.toPandas()

print(f"Total rows loaded: {len(df_pd):,}")
print(f"Columns: {list(df_pd.columns)}")

# Identify relevant columns
# The kh_asset table should have columns like:
#   - source text / extracted text (user input)
#   - generated JSON output (assistant response / ground truth)
#   - classification, confidence, etc.
print(f"\nColumn dtypes:")
print(df_pd.dtypes.to_string())

# Show sample values for key columns
print(f"\nSample of first row (non-null columns):")
first_row = df_pd.iloc[0]
for col in df_pd.columns:
    val = first_row[col]
    if val is not None and str(val).strip():
        preview = str(val)[:100]
        print(f"  {col}: {preview}{'...' if len(str(val)) > 100 else ''}")

# COMMAND ----------

# ============================================================
# Format as RFT (Reinforcement Fine-Tuning) JSONL for o4-mini
#
# Azure Foundry RFT format (NO system messages allowed):
# {
#   "messages": [
#     {"role": "user", "content": "<system prompt>\n\n<user content>"}
#   ],
#   "Asset": "...",
#   "Client_issue": "...",
#   "AB_IP_generated": "...",
#   "Tax_years": "...",
#   "References": ["ref1", "ref2"]
# }
#
# Key requirements:
# - System messages are NOT supported in reinforcement examples
# - The final message MUST be a user role
# - System prompt is merged into the user message
# - References must be a flat list of strings (no nested lists)
# ============================================================

import random
import re
import pandas as pd
import ast

# ══ COLUMN CONFIGURATION ══
SOURCE_PATH_COL = "redacted_adls_path"
APPROVAL_COL = "is_approved"

# Mapping: kh_asset column → RFT output field name
FIELD_MAP = {
    "Asset": "Asset",
    "Client_Issue": "Client_issue",
    "AB_Analysis": "AB_IP_generated",
    "Tax_Years": "Tax_years",
    "References": "References",
}

# ══ DATASET PARAMETERS ══
VALIDATION_SPLIT = 0.15
MIN_SOURCE_LENGTH = 100  # lowered to include shorter docs (need ≥11 examples)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Filter to rows where is_ip is set (True or False) — these are assets that have
# been through IP classification review. This gives ~14 rows vs 11 from is_approved alone.
# We also include any additional rows where is_approved is set but is_ip is null.
MIN_TRAIN_EXAMPLES = 10
IP_COL = "is_ip"

# Primary filter: is_ip IS NOT NULL (reviewed for IP classification)
df_ip_reviewed = df_pd[df_pd[IP_COL].notna()].copy()
print(f"Rows with is_ip set (True/False): {len(df_ip_reviewed):,}")
print(f"  - is_ip=True:  {(df_ip_reviewed[IP_COL] == True).sum():,}")
print(f"  - is_ip=False: {(df_ip_reviewed[IP_COL] == False).sum():,}")

# Also pick up any rows that have is_approved set but is_ip is null
df_approved_only = df_pd[
    df_pd[APPROVAL_COL].notna() &
    df_pd[IP_COL].isna()
].copy()
if len(df_approved_only) > 0:
    print(f"  + Adding {len(df_approved_only)} rows with is_approved set but is_ip null")

df_pd = pd.concat([df_ip_reviewed, df_approved_only], ignore_index=True).drop_duplicates(subset=["asset_id"])
print(f"  = Reviewed rows for RFT: {len(df_pd):,}")
print(f"    is_approved breakdown: True={( df_pd[APPROVAL_COL] == True).sum()}, False={(df_pd[APPROVAL_COL] == False).sum()}, NULL={df_pd[APPROVAL_COL].isna().sum()}")

# Azure requires at least 10 TRAINING examples + 1 validation = 11 minimum.
# Add unreviewed rows with populated Asset + redacted_adls_path to reach this threshold.
MIN_TOTAL_EXAMPLES_NEEDED = 16  # buffer: ~25% get skipped in build loop
current_count = len(df_pd)
if current_count < MIN_TOTAL_EXAMPLES_NEEDED:
    shortfall = MIN_TOTAL_EXAMPLES_NEEDED - current_count
    df_all_pd = spark.table("hive_metastore.default.kh_asset").toPandas()
    df_extra = df_all_pd[
        ~df_all_pd["asset_id"].isin(df_pd["asset_id"]) &
        df_all_pd["Asset"].notna() & (df_all_pd["Asset"].str.strip() != "") &
        df_all_pd[SOURCE_PATH_COL].notna() & (df_all_pd[SOURCE_PATH_COL].str.strip() != "")
    ].head(shortfall)
    print(f"  + Adding {len(df_extra)} extra rows (with Asset + redacted_adls_path) to reach minimum")
    df_pd = pd.concat([df_pd, df_extra], ignore_index=True).drop_duplicates(subset=["asset_id"])

print(f"  = Total rows for RFT: {len(df_pd):,}")

# ── Helper: extract analysis_results from asset_json ──
# (From cleanup_asset_json notebook — unwraps {ok, text: "<json>"} envelope)
def extract_analysis_results(response):
    """Extract clean analysis_results from the raw asset JSON response.
    Handles: {ok, text: '<json string>'} wrapper, markdown fences, flat objects."""
    if response is None or (isinstance(response, str) and not response.strip()):
        return []
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception:
            return []
    if not isinstance(response, dict):
        return []
    # Check for blocked/error
    if response.get("blocked") or response.get("error"):
        return []
    # Extract the text field (contains the actual JSON as a string)
    text = response.get("text", "")
    if not text:
        # Maybe it's already a flat result dict
        if "analysis_results" in response:
            ar = response["analysis_results"]
            return ar if isinstance(ar, list) else [ar]
        return []
    # Strip markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Parse inner JSON
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, dict) and "analysis_results" in parsed:
        results = parsed["analysis_results"]
    elif isinstance(parsed, list):
        results = parsed
    else:
        results = [parsed]
    return results if results else []

# ── Helper: clean bullet-point formatting from values ──
# (From cleanup_asset_json notebook)
def clean_value(v):
    """Remove bullet-point formatting (\n•, \n-, \n*) from strings.
    Recursively cleans lists and dicts."""
    if isinstance(v, str):
        v = re.sub(r'\n[\u2022\-\*]\s*', ' ', v)
        v = re.sub(r'^[\u2022\-\*]\s*', '', v)
        v = re.sub(r'\s+', ' ', v).strip()
        return v
    elif isinstance(v, list):
        return [clean_value(item) for item in v]
    elif isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}
    return v

# ── Helper: normalize References to flat list of strings ──
def normalize_references(value):
    """Normalize references into a flat list of strings.
    Handles: ["['CFC-regels']"] -> ["CFC-regels"]
             "['CFC-regels']"   -> ["CFC-regels"]
             ["CFC-regels"]     -> ["CFC-regels"]
    """
    if value is None:
        return []
    # If value is a string, wrap in list for uniform processing
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [str(value)]

    out = []
    for r in value:
        if r is None:
            continue
        if isinstance(r, str):
            s = r.strip()
            # Try to unwrap Python-style list string, e.g. "['CFC-regels']"
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if item is not None:
                                out.append(str(item).strip())
                        continue
                except Exception:
                    pass
                # Try JSON parse
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if item is not None:
                                out.append(str(item).strip())
                        continue
                except Exception:
                    pass
            if s:
                out.append(s)
        else:
            out.append(str(r).strip())

    # Dedupe while preserving order
    deduped = []
    seen = set()
    for item in out:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped

# ── Helper: merge system prompt into single user message ──
def merge_system_into_user(system_content, user_content):
    """Azure Foundry RFT rejects system messages.
    Merge system + user content into ONE user message.
    Result: [{"role": "user", "content": "<system>\n\n<user>"}]
    """
    parts = []
    if system_content and system_content.strip():
        parts.append(system_content.strip())
    if user_content and user_content.strip():
        parts.append(user_content.strip())
    merged = "\n\n".join(parts)
    if not merged:
        raise ValueError("Merged user content is empty")
    return [{"role": "user", "content": merged}]

# ── Pre-load source documents from ADLS (parquet format) ──
print("Reading source documents from ADLS (parquet format)...")
source_text_cache = {}
unique_paths = df_pd[SOURCE_PATH_COL].dropna().unique()
for adls_path in unique_paths:
    adls_path = str(adls_path).strip()
    if adls_path and adls_path not in source_text_cache:
        try:
            chunks_df = spark.read.parquet(adls_path)
            full_text = chunks_df.orderBy("chunk_index").select(
                F.concat_ws("\n", F.collect_list("redacted_text"))
            ).first()[0]
            source_text_cache[adls_path] = full_text
        except Exception:
            source_text_cache[adls_path] = None
print(f"  Loaded {len([v for v in source_text_cache.values() if v]):,} source documents from {len(unique_paths):,} paths")

# ── Build RFT examples ──
rft_examples = []
skipped = {
    "no_source_text": 0,
    "source_too_short": 0,
    "no_reference_fields": 0,
}

for _, row in df_pd.iterrows():
    # Get source text
    adls_path = str(row.get(SOURCE_PATH_COL, "") or "").strip()
    source_text = source_text_cache.get(adls_path) or ""
    if not source_text:
        skipped["no_source_text"] += 1
        continue
    if len(source_text) < MIN_SOURCE_LENGTH:
        skipped["source_too_short"] += 1
        continue

    # Build messages: merge system prompt into single user message
    user_content = f'Text to analyze:\n\"\"\"{source_text}\"\"\"'
    messages = merge_system_into_user(GENERATOR_SYSTEM_PROMPT, user_content)

    # Build reference fields from human-corrected columns
    # Apply clean_value() to strip bullet-point formatting (from cleanup_asset_json)
    ref_fields = {}
    for kh_col, rft_field in FIELD_MAP.items():
        val = row.get(kh_col)
        if rft_field == "References":
            ref_fields[rft_field] = clean_value(normalize_references(val))
        else:
            raw = str(val).strip() if val is not None and str(val).strip() else ""
            ref_fields[rft_field] = clean_value(raw)

    # Must have at least Asset or Client_issue
    if not ref_fields.get("Asset") and not ref_fields.get("Client_issue"):
        skipped["no_reference_fields"] += 1
        continue

    # Extract original model output from asset_json using extract_analysis_results
    # This is the raw model output BEFORE human correction
    asset_json_raw = row.get("asset_json", "")
    original_model_fields = {}
    if asset_json_raw and str(asset_json_raw).strip():
        results = extract_analysis_results(asset_json_raw)
        if results and isinstance(results[0], dict):
            ar = results[0]
            # Extract and clean original model output fields
            original_model_fields = {
                "original_model_Asset": clean_value(str(ar.get("Asset", ""))),
                "original_model_Client_issue": clean_value(str(ar.get("Client_issue", ar.get("Client_Issue", "")))),
                "original_model_AB_IP_generated": clean_value(str(ar.get("AB_IP_generated", ar.get("AB_Analysis", "")))),
            }

    # Add review metadata for semantic grader
    review_comment = str(row.get("review_comment") or "").strip()
    is_approved = row.get(APPROVAL_COL)
    approval_status = "approved" if is_approved == True else ("rejected" if is_approved == False else "unreviewed")

    # Build RFT example
    example = {
        "messages": messages,
        **ref_fields,
        **original_model_fields,
        "review_comment": review_comment,
        "approval_status": approval_status
    }
    rft_examples.append(example)

print(f"\n\u2705 Built {len(rft_examples):,} RFT training examples")
print(f"   Skipped breakdown:")
for reason, count in skipped.items():
    print(f"     {reason}: {count:,}")

# ── Shuffle and split ──
# Azure OpenAI reinforcement fine-tuning REQUIRES a validation file.
# We must ALWAYS have at least 1 validation example.
random.shuffle(rft_examples)

# Azure RFT requires: at least 10 TRAINING examples + at least 1 VALIDATION example
MIN_TRAIN_REQUIRED = 10

if len(rft_examples) < MIN_TRAIN_REQUIRED:
    raise ValueError(
        f"Need at least {MIN_TRAIN_REQUIRED} examples for Azure RFT, "
        f"but only got {len(rft_examples)}.\n"
        f"Add more rows with populated Asset + redacted_adls_path to kh_asset."
    )

# Split strategy:
# - If we have > MIN_TRAIN_REQUIRED: split normally (train gets bulk, val gets remainder)
# - If we have exactly MIN_TRAIN_REQUIRED: use ALL for training, duplicate 1 for validation
#   (Azure requires a validation file for RFT; at this scale it's just for loss monitoring)
if len(rft_examples) > MIN_TRAIN_REQUIRED:
    split_idx = max(MIN_TRAIN_REQUIRED, min(len(rft_examples) - 1, int(len(rft_examples) * (1 - VALIDATION_SPLIT))))
    train_data = rft_examples[:split_idx]
    val_data = rft_examples[split_idx:]
else:
    # Exactly 10: use all for training, duplicate 1 for validation
    train_data = rft_examples
    val_data = [rft_examples[0]]  # duplicate first example for validation
    print(f"   ⚠️ Exactly {MIN_TRAIN_REQUIRED} examples — using all for training, duplicating 1 for validation")

# Safety assertions
assert len(train_data) >= MIN_TRAIN_REQUIRED, f"Train set has {len(train_data)} examples, need at least {MIN_TRAIN_REQUIRED}"
assert len(val_data) >= 1, "Validation set must have at least 1 example for Azure RFT"

print(f"\n   Train set: {len(train_data):,} examples")
print(f"   Validation set: {len(val_data):,} examples")
print(f"   Split ratio: {len(train_data)}/{len(val_data)} ({100*(1-VALIDATION_SPLIT):.0f}%/{100*VALIDATION_SPLIT:.0f}%)")

# ── Preview one example ──
print(f"\n{'='*60}")
print(f"SAMPLE RFT EXAMPLE (messages format + reference fields):")
print(f"{'='*60}")
if rft_examples:
    sample = rft_examples[0]
    msg = sample['messages'][0]
    print(f"  messages: [{{role: '{msg['role']}', content: ...{len(msg['content'])} chars...}}]")
    print(f"  (No system message — merged into user)")
    for k, v in sample.items():
        if k != "messages":
            preview = str(v)[:200]
            print(f"  {k}: {preview}{'...' if len(str(v)) > 200 else ''}")

# COMMAND ----------

# ============================================================
# Save Train and Validation JSONL files to ADLS
# Output: stage/fine-tuning/synthetic/
#   - train.jsonl
#   - validation.jsonl
# ============================================================

import json
from datetime import datetime, timezone

# Output paths
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
output_folder = f"fine-tuning/dpo/{timestamp}"
output_base = ABFSS("stage", output_folder)

train_path = f"{output_base}/train.jsonl"
val_path = f"{output_base}/validation.jsonl"

# Write train.jsonl
train_content = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in train_data)
dbutils.fs.put(train_path, train_content, overwrite=True)
print(f"\u2705 Saved train.jsonl: {len(train_data):,} examples")
print(f"   Path: {train_path}")
print(f"   Size: {len(train_content):,} chars")

# Write validation.jsonl — REQUIRED for Azure OpenAI reinforcement fine-tuning
if not val_data:
    raise RuntimeError(
        "\u274c val_data is empty! Azure RFT requires a validation file.\n"
        "Re-run Cell 13 first to regenerate the train/val split."
    )
val_content = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in val_data)
dbutils.fs.put(val_path, val_content, overwrite=True)
print(f"\n\u2705 Saved validation.jsonl: {len(val_data):,} examples")
print(f"   Path: {val_path}")
print(f"   Size: {len(val_content):,} chars")

# Summary
print(f"\n{'='*60}")
print(f"SYNTHETIC DATASET GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Output folder: {output_base}/")
print(f"  Train examples: {len(train_data):,}")
print(f"  Validation examples: {len(val_data):,}")
print(f"  System prompt: Asset_Creation_Tax_V7.4.10 ({len(GENERATOR_SYSTEM_PROMPT):,} chars)")
print(f"  Format: Azure OpenAI DPO Fine-Tuning JSONL")
print(f"  Structure: input (system+user) / preferred_output / non_preferred_output")
print(f"\n  \u2139\ufe0f Next steps:")
print(f"  1. Upload train.jsonl and validation.jsonl to Azure OpenAI Studio")
print(f"  2. Create a DPO fine-tuning job (model: gpt-4o or gpt-4o-mini)")
print(f"  3. Select 'Direct Preference Optimization' as the training method")
print(f"  4. Monitor training metrics in Azure Portal")

# COMMAND ----------

# MAGIC %pip install openai sendgrid --quiet

# COMMAND ----------

# ============================================================
# Submit o4-mini Reinforcement Fine-Tuning Job
# via Azure OpenAI Python SDK + REST API for job creation
# Model: o4-mini-2025-04-16
# Method: Reinforcement
# Training type: Global Standard
# ============================================================

from openai import AzureOpenAI
import os
import requests
import json
import time

# ------------------------------------------------------------
# Secrets / configuration
# ------------------------------------------------------------
storage_account_name = dbutils.secrets.get(
    scope="keyvault-secret-scope",
    key="Storage-Account-Name"
)

AZURE_OPENAI_ENDPOINT = "https://AB-openai.openai.azure.com/"

# IMPORTANT:
# Do NOT hardcode the key. Store it in Key Vault / Databricks secret scope.

AZURE_OPENAI_KEY = ""

FILE_API_VERSION = "2025-04-01-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=FILE_API_VERSION
)

# ------------------------------------------------------------
# Step 1: Copy JSONL files from DBFS to local /tmp without truncation
# ------------------------------------------------------------
train_path = f"{output_base}/train.jsonl"
val_path = f"{output_base}/validation.jsonl"

local_train_fs = "/tmp/train.jsonl"
local_val_fs = "/tmp/validation.jsonl"

print(f"Copying train from: {train_path}")
dbutils.fs.cp(train_path, "file:/tmp/train.jsonl", True)

# Validation file is REQUIRED for Azure OpenAI reinforcement fine-tuning
has_val = False
try:
    dbutils.fs.ls(val_path)
    has_val = True
    print(f"Copying val from:   {val_path}")
    dbutils.fs.cp(val_path, "file:/tmp/validation.jsonl", True)
except Exception:
    raise RuntimeError(
        f"\u274c Validation file not found at {val_path}\n"
        f"Azure OpenAI reinforcement fine-tuning REQUIRES a validation dataset.\n"
        f"Re-run Cell 13 to generate train/val split (must have ≥2 examples)."
    )

# Sanity check: validate first JSONL line
with open(local_train_fs, "r", encoding="utf-8") as f:
    first_line = f.readline()
    json.loads(first_line)
print("\u2705 First training JSONL line is valid JSON")

if has_val:
    with open(local_val_fs, "r", encoding="utf-8") as f:
        first_line = f.readline()
        json.loads(first_line)
    print("\u2705 First validation JSONL line is valid JSON")

# ------------------------------------------------------------
# Step 2: Upload files
# ------------------------------------------------------------
with open(local_train_fs, "rb") as f:
    train_file = client.files.create(file=f, purpose="fine-tune")
print(f"\u2705 Training file uploaded: {train_file.id}")

val_file = None
if has_val:
    with open(local_val_fs, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"\u2705 Validation file uploaded: {val_file.id}")


# ------------------------------------------------------------
# Step 3: Wait for files to be processed
# ------------------------------------------------------------
def wait_for_processed(client, file_id):
    while True:
        file_obj = client.files.retrieve(file_id)
        status = getattr(file_obj, "status", None)

        print(f"{file_id} status: {status}")

        if status in ("processed", "succeeded", "completed"):
            return file_obj

        if status in ("failed", "error"):
            raise RuntimeError(f"File {file_id} failed processing: {file_obj}")

        time.sleep(5)


wait_for_processed(client, train_file.id)
if val_file:
    wait_for_processed(client, val_file.id)

print("\u2705 All files processed")


# ------------------------------------------------------------
# Step 4: Multi-Grader Architecture
# (1) Structural Python grader (weight 0.15)
#     - Uses extract_analysis_results pattern from cleanup_asset_json
#     - Validates JSON structure, required fields, references, tax years
#     - No file I/O, no external dependencies
# (2) Semantic score_model grader (weight 0.85, gpt-4o)
#     - Compares model output (Asset, Client_issue, AB_IP_generated)
#       vs human-corrected columns (Asset, Client_Issue, AB_Analysis)
#     - Uses review_comments and approval_status for reward
# ------------------------------------------------------------

# ══ STRUCTURAL GRADER (Python) ══
STRUCTURAL_GRADER_SOURCE = r'''
def grade(sample, item):
    import json
    import re

    def safe_parse(raw):
        """3-fallback JSON extraction (from cleanup_asset_json pattern)."""
        if raw is None:
            return None
        if isinstance(raw, dict):
            # Handle {ok, text: "<json>"} wrapper
            text_val = raw.get("text")
            if isinstance(text_val, str):
                return safe_parse(text_val)
            return raw

        raw = str(raw).strip()
        if not raw:
            return None

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # Attempt 1: direct json.loads
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Attempt 2: brace-balancing (safe, no regex backtracking risk)
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except Exception:
                        return None
        return None

    def extract_analysis_result(parsed):
        """Navigate wrappers to get first analysis result dict."""
        if not isinstance(parsed, dict):
            return None
        # Handle {ok, text: "..."} wrapper
        if "text" in parsed and isinstance(parsed.get("text"), str):
            inner = safe_parse(parsed["text"])
            if isinstance(inner, dict):
                return extract_analysis_result(inner)
            return None
        # Handle {analysis_results: [...]} wrapper
        ar = parsed.get("analysis_results")
        if isinstance(ar, list) and len(ar) > 0 and isinstance(ar[0], dict):
            return ar[0]
        # Flat object with expected fields
        return parsed

    try:
        # Get model output — ONLY use output_text to avoid FileNotFoundError
        # in Azure sandbox (output_json may not exist as an attribute)
        try:
            output_text = sample.output_text if hasattr(sample, "output_text") else ""
        except Exception:
            output_text = ""
        if not output_text:
            try:
                output_text = sample.get("output_text", "") if hasattr(sample, "get") else ""
            except Exception:
                output_text = ""
        if not output_text:
            return 0.0

        parsed = safe_parse(output_text)
        if not parsed or not isinstance(parsed, dict):
            return 0.0  # Unparseable

        data = extract_analysis_result(parsed)
        if not isinstance(data, dict) or not data:
            return 0.01  # JSON valid but no KH fields

        # Score components
        score = 0.0

        # (1) JSON parseable: 0.30
        score += 0.30

        # (2) Required fields present: 0.30 * (N/5)
        # Signal: if NO fields found at all -> return 0.01 (valid JSON, no KH content)
        FIELD_ALIASES = {
            "Asset": ["Asset"],
            "Client_issue": ["Client_issue", "Client_Issue", "client_issue"],
            "AB_IP_generated": ["AB_IP_generated", "AB_Analysis", "AB_ip_generated"],
            "Tax_years": ["Tax_years", "Tax_Years", "tax_years"],
            "References": ["References", "references"],
        }
        fields_found = 0
        for canonical, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                val = data.get(alias)
                if val not in (None, "", [], {}):
                    fields_found += 1
                    break
        if fields_found == 0:
            return 0.01  # JSON valid but no KH fields
        score += 0.30 * (fields_found / 5.0)

        # (3) References non-empty list: 0.20
        refs = None
        for alias in ["References", "references"]:
            refs = data.get(alias)
            if refs is not None:
                break
        if isinstance(refs, list) and len(refs) > 0:
            score += 0.20
        elif isinstance(refs, str) and refs.strip():
            score += 0.10

        # (4) Tax year valid (regex 20[0-3]\d): 0.20
        tax_year_val = ""
        for alias in ["Tax_years", "Tax_Years", "tax_years"]:
            tv = data.get(alias)
            if tv:
                tax_year_val = str(tv)
                break
        if re.search(r"20[0-3]\d", tax_year_val):
            score += 0.20
        elif tax_year_val.strip():
            score += 0.05

        return round(min(max(score, 0.0), 1.0), 2)

    except Exception:
        return 0.0
    except BaseException:
        return 0.0
'''

# ══ SEMANTIC GRADER (gpt-4o score_model) ══
SEMANTIC_GRADER_INPUT = [
    {
        "role": "system",
        "content": """You are an expert Dutch/international tax law reviewer evaluating the quality of a model-generated tax asset against a human-corrected reference.

You will receive:
1. The model's generated output (Asset, Client_issue, AB_IP_generated)
2. The human-corrected reference (same fields, after expert review)
3. The reviewer's approval status and comments

Score the model output on a scale of 0.0 to 1.0 based on these dimensions:

- Classification Accuracy (35%): Does the model's Asset name and Client_issue correctly identify the tax issue type and key facts?
- Factuality (25%): Does the model's AB_IP_generated contain factual analysis without hallucination compared to the reference?
- Completeness (20%): Are tax years, references, and edge cases covered as in the reference?
- Professional Quality (10%): Is the output framed in proper tax-domain language with clarity?
- Reviewer Alignment (10%): Does the model output address the concerns raised in the reviewer's comments?

Penalties:
- Wrong classification: -0.35
- Hallucinated facts not in reference: -0.30
- Missing references that appear in human version: -0.25
- Ignoring reviewer feedback (if reviewer rejected and gave specific comments): -0.20

Return ONLY a decimal number between 0.0 and 1.0."""
    },
    {
        "role": "user",
        "content": """Reviewer Status: {{item.approval_status}}
Reviewer Comments: {{item.review_comment}}

--- HUMAN-CORRECTED REFERENCE (ground truth) ---
Asset: {{item.Asset}}
Client_issue: {{item.Client_issue}}
AB_IP_generated: {{item.AB_IP_generated}}
Tax_years: {{item.Tax_years}}
References: {{item.References}}

--- ORIGINAL MODEL OUTPUT (before human correction, extracted from asset_json) ---
Asset: {{item.original_model_Asset}}
Client_issue: {{item.original_model_Client_issue}}
AB_IP_generated: {{item.original_model_AB_IP_generated}}

--- NEW MODEL OUTPUT (to be scored) ---
{{sample.output_text}}"""
    }
]

# ------------------------------------------------------------
# Step 4b: Validate structural grader locally
# ------------------------------------------------------------
ns = {}
exec(STRUCTURAL_GRADER_SOURCE, ns)
grade_fn = ns["grade"]

# Test cases
test_cases = [
    # (description, sample, expected_min, expected_max)
    ("Perfect output",
     {"output_text": '{"analysis_results":[{"Asset":"a","Client_issue":"b","AB_IP_generated":"c","Tax_years":"2024","References":["Art 10"]}]}'},
     {}, 1.0, 1.0),
    ("Unparseable garbage",
     {"output_text": "not json at all"}, {}, 0.0, 0.0),
    ("JSON but no KH fields",
     {"output_text": '{"foo": "bar"}'}, {}, 0.01, 0.01),
    ("Wrapped in {ok, text}",
     {"output_text": json.dumps({"ok": True, "text": json.dumps({"analysis_results": [{"Asset": "x", "Client_issue": "y", "AB_IP_generated": "z", "Tax_years": "2025", "References": ["ref1"]}]})})},
     {}, 0.9, 1.0),
    ("Missing refs and tax_years",
     {"output_text": '{"Asset":"a","Client_issue":"b","AB_IP_generated":"c"}'}, {}, 0.3, 0.7),
    ("Markdown fenced",
     {"output_text": '```json\n{"Asset":"a","Client_issue":"b","AB_IP_generated":"c","Tax_years":"2024","References":["r"]}\n```'},
     {}, 0.9, 1.0),
]

print("Structural grader validation:")
all_passed = True
for desc, sample, item, exp_min, exp_max in test_cases:
    score = grade_fn(sample, item)
    passed = exp_min <= score <= exp_max
    status = "\u2705" if passed else "\u274c"
    print(f"  {status} {desc}: {score:.2f} (expected {exp_min:.2f}-{exp_max:.2f})")
    if not passed:
        all_passed = False

assert all_passed, "Structural grader validation FAILED"
print(f"\n\u2705 All structural grader tests passed")
print(f"\u2705 Semantic grader input configured (gpt-4o score_model)")

# ------------------------------------------------------------
# Step 5: Create fine-tuning job
# ------------------------------------------------------------
base_url = AZURE_OPENAI_ENDPOINT.rstrip("/")
headers = {
    "api-key": AZURE_OPENAI_KEY,
    "Content-Type": "application/json"
}

api_ver = "2025-04-01-preview"
url = f"{base_url}/openai/fine_tuning/jobs?api-version={api_ver}"

payload = {
    "model": "o4-mini-2025-04-16",
    "training_file": train_file.id,
    "suffix": "kh-tax-asset",
    "trainingType": "globalstandard",
    "method": {
        "type": "reinforcement",
        "reinforcement": {
            "hyperparameters": {
                "n_epochs": 2,
                "compute_multiplier": 1.0
            },
            "grader": {
                "type": "multi",
                "graders": {
                    "structural": {
                        "type": "python",
                        "name": "kh-structural-grader",
                        "source": STRUCTURAL_GRADER_SOURCE
                    },
                    "semantic": {
                        "type": "score_model",
                        "name": "kh-semantic-grader",
                        "model": "gpt-4o",
                        "input": SEMANTIC_GRADER_INPUT,
                        "range": [0, 1]
                    }
                },
                "calculate_output": "0.15 * structural + 0.85 * semantic"
            }
        }
    }
}

# Validation file is REQUIRED for reinforcement fine-tuning
if not val_file:
    raise RuntimeError("val_file is None — Azure RFT requires a validation file. Re-run cells 13+14.")
payload["validation_file"] = val_file.id

print(f"Submitting reinforcement fine-tuning job to {api_ver}...")
print(json.dumps({
    "model": payload["model"],
    "training_file": payload["training_file"],
    "validation_file": payload.get("validation_file"),
    "suffix": payload["suffix"],
    "trainingType": payload["trainingType"],
    "method_type": payload["method"]["type"],
    "grader_type": payload["method"]["reinforcement"]["grader"]["type"],
    "graders": list(payload["method"]["reinforcement"]["grader"]["graders"].keys()),
    "formula": payload["method"]["reinforcement"]["grader"]["calculate_output"]
}, indent=2))

resp = requests.post(url, headers=headers, json=payload)

try:
    resp_json = resp.json()
except Exception:
    resp_json = {"raw": resp.text}

if resp.status_code in (200, 201):
    job_response = resp_json
    print("\n\u2705 o4-mini reinforcement fine-tuning job created!")
    print(f"   Job ID: {job_response.get('id')}")
    print(f"   Status: {job_response.get('status')}")
    print(f"   Model: {job_response.get('model')}")
else:
    print(f"\n\u274c {resp.status_code}")
    print(json.dumps(resp_json, indent=2))
    print("\nFiles uploaded:")
    print(f"   Train: {train_file.id}")
    print(f"   Val:   {val_file.id if val_file else 'None (no validation file)'}")

# COMMAND ----------

# ============================================================
# Check status of o4-mini fine-tuning job
# ============================================================
import requests
import json

AZURE_OPENAI_ENDPOINT = "https://eyi-kh-openai.openai.azure.com/"
AZURE_OPENAI_KEY = ""

job_id = "ftjob-a625d8c2686c4cacadc9596e88be453b"
api_ver = "2025-04-01-preview"

url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/fine_tuning/jobs/{job_id}?api-version={api_ver}"
headers = {"api-key": AZURE_OPENAI_KEY}

resp = requests.get(url, headers=headers)

if resp.status_code == 200:
    job = resp.json()
    print(f"Job ID:        {job.get('id')}")
    print(f"Status:        {job.get('status')}")
    print(f"Model:         {job.get('model')}")
    print(f"Created at:    {job.get('created_at')}")
    print(f"Fine-tuned model: {job.get('fine_tuned_model', 'N/A (still training)')}")
    
    # Show error if failed
    error = job.get('error')
    if error:
        print(f"\n\u274c Error: {json.dumps(error, indent=2)}")
    
    # Show training metrics if available
    result_files = job.get('result_files', [])
    if result_files:
        print(f"\nResult files: {result_files}")
else:
    print(f"\u274c {resp.status_code}: {resp.text[:500]}")

# COMMAND ----------

# ============================================================
# Poll fine-tuning job and send email when training completes
# Uses SendGrid (same pattern as email_asset_approval_v3)
# ============================================================
import requests
import json
import time
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Subject, HtmlContent, PlainTextContent

# ── Configuration ──
SENDGRID_API_KEY = dbutils.secrets.get(scope="keyvault-secret-scope", key="SendGrid-API-Key")
AZURE_OPENAI_ENDPOINT = "https://eyi-kh-openai.openai.azure.com/"
AZURE_OPENAI_KEY = ""

SENDER_EMAIL = "noreply-ab@email.com"
SENDER_NAME = "Aswin Bhaskaran"
NOTIFY_EMAILS = ["aswin.bhaskaran@ab.com"]  # TODO: add recipients

JOB_ID = "ftjob-a625d8c2686c4cacadc9596e88be453b"
API_VERSION = "2025-04-01-preview"
POLL_INTERVAL_SECONDS = 120  # Check every 2 minutes
MAX_POLL_ATTEMPTS = 180  # Max ~6 hours of polling

# ── Poll job status ──
def get_job_status(job_id):
    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/fine_tuning/jobs/{job_id}?api-version={API_VERSION}"
    headers = {"api-key": AZURE_OPENAI_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    return None

# ── Send email notification ──
def send_training_notification(job_data, status):
    model = job_data.get("model", "o4-mini")
    fine_tuned_model = job_data.get("fine_tuned_model", "N/A")
    job_id = job_data.get("id", JOB_ID)
    error_msg = job_data.get("error", {}).get("message", "") if job_data.get("error") else ""

    if status == "succeeded":
        subject_text = f"\u2705 Fine-Tuning Complete: {model} → {fine_tuned_model}"
        body_html = f"""
        <html><body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2E7D32;">\u2705 Fine-Tuning Job Completed Successfully</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px; font-weight: bold;">Job ID:</td><td style="padding: 8px;">{job_id}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Base Model:</td><td style="padding: 8px;">{model}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Fine-Tuned Model:</td><td style="padding: 8px;"><strong>{fine_tuned_model}</strong></td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Training Type:</td><td style="padding: 8px;">Reinforcement (Global)</td></tr>
        </table>
        <p style="margin-top: 16px;">The model is now ready for deployment in Azure OpenAI Studio.</p>
        <p style="color: #666; font-size: 12px;">Sent by Knowledge Harvester - Fine-Tuning Pipeline</p>
        </body></html>
        """
    else:
        subject_text = f"\u274c Fine-Tuning Failed: {model} (Job: {job_id[-12:]})"
        body_html = f"""
        <html><body style="font-family: Arial, sans-serif;">
        <h2 style="color: #C62828;">\u274c Fine-Tuning Job Failed</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px; font-weight: bold;">Job ID:</td><td style="padding: 8px;">{job_id}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Base Model:</td><td style="padding: 8px;">{model}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Status:</td><td style="padding: 8px; color: red;">FAILED</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Error:</td><td style="padding: 8px;">{error_msg or 'No details available'}</td></tr>
        </table>
        <p style="margin-top: 16px;">Please check the Azure OpenAI Portal for full error details.</p>
        <p style="color: #666; font-size: 12px;">Sent by Knowledge Harvester - Fine-Tuning Pipeline</p>
        </body></html>
        """

    # Send via SendGrid
    sg_client = SendGridAPIClient(api_key=SENDGRID_API_KEY)
    message = Mail(
        from_email=Email(SENDER_EMAIL, SENDER_NAME),
        to_emails=[To(email) for email in NOTIFY_EMAILS],
        subject=Subject(subject_text),
        html_content=HtmlContent(body_html)
    )

    try:
        response = sg_client.client.mail.send.post(request_body=message.get())
        if response.status_code in (200, 201, 202):
            print(f"\u2705 Notification email sent to: {', '.join(NOTIFY_EMAILS)}")
        else:
            print(f"\u274c Email send failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"\u274c Email send error: {e}")

# ── Main: Poll and notify ──
print(f"Polling fine-tuning job: {JOB_ID}")
print(f"Checking every {POLL_INTERVAL_SECONDS}s (max {MAX_POLL_ATTEMPTS} attempts)...\n")

for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
    job_data = get_job_status(JOB_ID)
    if not job_data:
        print(f"  [{attempt}] Could not fetch job status. Retrying...")
        time.sleep(POLL_INTERVAL_SECONDS)
        continue

    status = job_data.get("status", "unknown")
    print(f"  [{attempt}] Status: {status}")

    if status == "succeeded":
        print(f"\n\u2705 Training complete! Fine-tuned model: {job_data.get('fine_tuned_model')}")
        send_training_notification(job_data, "succeeded")
        break
    elif status in ("failed", "cancelled"):
        print(f"\n\u274c Training {status}.")
        error = job_data.get("error")
        if error:
            print(f"   Error: {json.dumps(error, indent=2)}")
        send_training_notification(job_data, "failed")
        break
    else:
        # Still running (pending, running, validating_files, etc.)
        time.sleep(POLL_INTERVAL_SECONDS)

else:
    print(f"\n\u26a0\ufe0f Max poll attempts reached ({MAX_POLL_ATTEMPTS}). Job still running.")
    print(f"   Run Cell 17 manually to check status later.")