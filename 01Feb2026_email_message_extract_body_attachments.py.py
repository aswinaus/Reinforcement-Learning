# Databricks notebook source
| Format        | Reality                                       |
| ------------- | --------------------------------------------- |
| `.eml`        | Open RFC-822 standard (easy to parse)         |
| `.msg`        | **Proprietary Outlook binary**                |
| `.eml → .msg` | Do not do this - Requires Outlook COM / Redemption / Windows |
| Databricks    | Do not do this No Outlook, no COM, Linux-only              |


# COMMAND ----------

# MAGIC %pip install extract_msg
# MAGIC

# COMMAND ----------

/mnt/emails/raw/            # .msg + .eml files
/mnt/emails/attachments/   # extracted attachments


# COMMAND ----------

# .msg extractor
import extract_msg
import os
import uuid

def extract_from_msg(msg_path, attachment_root):
    msg = extract_msg.Message(msg_path)
    msg.process()

    body = msg.htmlBody or msg.body

    email_id = str(uuid.uuid4())
    attach_dir = os.path.join(attachment_root, email_id)
    os.makedirs(attach_dir, exist_ok=True)

    attachments = []

    for att in msg.attachments:
        # Skip inline images if desired
        if att.cid:
            continue

        fname = att.longFilename or att.shortFilename or f"attachment_{uuid.uuid4()}"
        fname = fname.encode("utf-8", "ignore").decode()

        save_path = os.path.join(attach_dir, fname)
        att.save(customPath=save_path)

        attachments.append({
            "attachment_name": fname,
            "attachment_path": save_path,
            "attachment_size": os.path.getsize(save_path)
        })

    msg.close()

    return body, attachments


# COMMAND ----------

# .eml extractor
from email import policy
from email.parser import BytesParser

def extract_from_eml(eml_path, attachment_root):
    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    body_part = msg.get_body(preferencelist=("html", "plain"))
    body = body_part.get_content() if body_part else None

    email_id = str(uuid.uuid4())
    attach_dir = os.path.join(attachment_root, email_id)
    os.makedirs(attach_dir, exist_ok=True)

    attachments = []

    for part in msg.iter_attachments():
        fname = part.get_filename()
        if not fname:
            continue

        save_path = os.path.join(attach_dir, fname)
        with open(save_path, "wb") as af:
            af.write(part.get_payload(decode=True))

        attachments.append({
            "attachment_name": fname,
            "attachment_path": save_path,
            "attachment_size": os.path.getsize(save_path)
        })

    return body, attachments


# COMMAND ----------

# unified processing pipeline
from pyspark.sql import Row

RAW_DIR = "/mnt/emails/raw"
ATTACH_DIR = "/dbfs/mnt/emails/attachments"

email_rows = []
attachment_rows = []

files = dbutils.fs.ls(RAW_DIR)

for f in files:
    path = f"/dbfs{f.path}"
    name = f.name.lower()

    try:
        if name.endswith(".msg"):
            body, attachments = extract_from_msg(path, ATTACH_DIR)
        elif name.endswith(".eml"):
            body, attachments = extract_from_eml(path, ATTACH_DIR)
        else:
            continue

        email_id = os.path.basename(path)

        email_rows.append(Row(
            email_id=email_id,
            file_name=f.name,
            body=body
        ))

        for a in attachments:
            attachment_rows.append(Row(
                email_id=email_id,
                attachment_name=a["attachment_name"],
                attachment_path=a["attachment_path"],
                attachment_size=a["attachment_size"]
            ))

    except Exception as e:
        print(f"Failed processing {f.name}: {e}")


# COMMAND ----------

# Convert to spark dataframes
emails_df = spark.createDataFrame(email_rows)
attachments_df = spark.createDataFrame(attachment_rows)

emails_df.display()
attachments_df.display()


# COMMAND ----------

# Delta Tables

emails_df.write.mode("append").format("delta").saveAsTable("emails")
attachments_df.write.mode("append").format("delta").saveAsTable("email_attachments")
