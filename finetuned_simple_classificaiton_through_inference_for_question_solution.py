# Databricks notebook source
# ==========================================================
# 03_semantic_problem_solution_classifier_native.py
# ==========================================================
# Purpose:
#   - Simple test: Input problem, solution, and tax year text as prompt, compute semantic score, print result.
#   - Debug: Show why a non-tax text chunk may be classified as "Tax Problem" or "Tax Solution" if score > threshold.
# ==========================================================

import requests
import numpy as np

# Databricks Model Serving endpoint
SERVING_ENDPOINT = "https://adb-3249086852123311.11.azuredatabricks.net/serving-endpoints/finetuned_model_inference/invocations"
DATABRICKS_TOKEN = ""

def get_embeddings_serving(texts):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": [f"passage: {t}" for t in texts]
    }
    response = requests.post(SERVING_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict):
        for key in ["predictions", "embeddings"]:
            if key in result:
                return result[key]
        if len(result) == 1:
            return list(result.values())[0]
        raise ValueError(f"Unknown response structure: {result}")
    return result

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def classify_semantic_tax_chunk(test_text, threshold=0.85, debug=False):
    prompts = {
        "Tax Problem": "This document describes a tax problem only or tax issue only or ta",
        "Tax Solution": "This document provides a tax solution or tax resolution.",
        "Tax Topic": "This document discusses a tax topic or refers to a specific tax type.",
        "Tax Year": "This document semantically refers to a tax year or the meaning of a tax year."
    }
    embeddings = {k: np.array(get_embeddings_serving([v])[0]) for k, v in prompts.items()}
    emb_test = np.array(get_embeddings_serving([test_text])[0])
    similarities = {k: cosine_similarity(emb_test, emb) for k, emb in embeddings.items()}
    print(f"Semantic scores: {similarities}")
    minus_one_count = sum(1 for score in similarities.values() if score == -1)
    if minus_one_count >= 3:
        print("Classification: Other")
    else:
        is_problem = similarities["Tax Problem"] >= threshold
        is_solution = similarities["Tax Solution"] >= threshold
        if is_problem or is_solution:
            print("Classification: Tax Problem Solution")        
        else:
            best_label = max(similarities, key=similarities.get)
            best_score = similarities[best_label]
            if best_score >= threshold:
                print(f"Classification: {best_label}")
            else:
                print("Classification: Other")
    for label, score in similarities.items():
        print(f"{label} similarity: {score:.4f}")

# Example usage:
test_prompt = "however, that Member States are prevented from enacting legal presumptions in their national law,\nprovided that they are specific and rest on sufficient grounds.\n98.      In the present case, Article 10a(1)(c) of the Law on Corporation Tax, and the obligation for the\ntax payer to justify that the arrangement in question is genuine, apply, in principle, only in cases where\nan intra-group loan has been concluded by a taxable entity with a related entity established in another\nMember, in which the interest charges collected by the latter are not taxed, or not taxed at a reasonable\nrate.  (55) Those specific circumstances can legitimately be regarded as indications of conduct that\nmight amount to abusive tax evasion, justifying a reversal of the burden of proof. (56)\n99.      Once the national tax authorities have determined that such an intra-group loan falls within the\nscope of that provision and, thus, might have been concluded for tax avoidance purposes, it is no"

classify_semantic_tax_chunk(test_prompt, threshold=0.80, debug=True)