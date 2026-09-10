"""
LLM explanation and Q&A logic for Project 1, Supply Chain.

Uses Google's Gemini API through its OpenAI-compatible endpoint. This is a
first-party API (not routed through a third-party aggregator), with a
genuinely free, permanent rate-limited tier, no payment method required.
The model never makes the risk decision itself, it only explains numbers
the model and rule-based logic in data_prep.py already computed.

Setup:
Set a GEMINI_API_KEY, either as an environment variable when running
locally, or as a Streamlit secret when deployed. Get a key for free at
aistudio.google.com. Never hardcode the key in this file or commit it
to GitHub.

Locally, create a file called .streamlit/secrets.toml (not committed to
GitHub) containing:
    GEMINI_API_KEY = "your-key-here"

On Streamlit Community Cloud, add the same key under your app's Settings,
Secrets, once deployed.
"""

import os
import requests
import streamlit as st

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Flash is the model Google's free tier covers with the best limits.
# Pro-tier Gemini models require billing enabled, Flash does not.
DEFAULT_MODEL = "gemini-3.6-flash"


def get_api_key():
    """Read the API key from Streamlit secrets first, then environment variables."""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")


def call_openrouter(messages, model=DEFAULT_MODEL, max_tokens=2000):
    """Send a chat request to Gemini and return the model's reply as text."""
    api_key = get_api_key()
    if not api_key:
        return (
            "No Gemini API key found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml locally, or to your app's secrets "
            "once deployed on Streamlit Community Cloud. Get a free key "
            "at aistudio.google.com."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "temperature": 1.0,
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            error_detail = _extract_error_message(response)
            return f"Gemini returned an error ({response.status_code}): {error_detail}"
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Could not reach the language model right now. Details: {e}"
    except (KeyError, IndexError):
        return "Received an unexpected response from the language model."


def _extract_error_message(response):
    """
    Pull a readable error message out of a failed response, regardless of
    which error shape the provider used. Google's endpoints sometimes wrap
    errors in a list instead of the plain object the OpenAI format uses,
    so this handles both instead of assuming one specific structure.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text

    if isinstance(body, list) and body:
        body = body[0]

    if isinstance(body, dict):
        error_obj = body.get("error", body)
        if isinstance(error_obj, dict):
            return error_obj.get("message", response.text)
        return str(error_obj)

    return response.text


def generate_explanation(sku_row):
    """
    Turn one SKU's computed numbers into a plain-English explanation.
    sku_row is a dict with keys like sku_id, category, closing_stock,
    forecasted_demand, days_of_cover, lead_time_days, risk_flag.
    """
    prompt = (
        "You are explaining a supply chain risk flag to a non-technical business user. "
        "Use only the numbers given below. Do not invent any numbers. Keep it to 2 or 3 "
        "plain sentences, no jargon, no bullet points.\n\n"
        f"SKU: {sku_row.get('sku_id')}\n"
        f"Category: {sku_row.get('category')}\n"
        f"Current stock: {round(sku_row.get('closing_stock', 0), 2)}\n"
        f"Forecasted daily demand: {round(sku_row.get('forecasted_demand', 0), 2)}\n"
        f"Days of cover (how many days current stock will last): {round(sku_row.get('days_of_cover', 0), 2)}\n"
        f"Lead time to get new stock (days): {sku_row.get('lead_time_days')}\n"
        f"Risk flag assigned: {sku_row.get('risk_flag')}\n\n"
        "Explain why this SKU received this specific flag, referencing the actual numbers above."
    )

    messages = [
        {"role": "system", "content": "You explain supply chain data clearly and simply, grounded only in the numbers you are given."},
        {"role": "user", "content": prompt},
    ]
    return call_openrouter(messages)


def answer_question(question, all_flags_df):
    """
    Answer a free-text question, grounded in the current flags table.
    Finds relevant rows if the question mentions a specific SKU, otherwise
    provides the model with a summary of all flagged SKUs.
    """
    question_lower = question.lower()
    mentioned_skus = [
        sku for sku in all_flags_df["sku_id"].astype(str)
        if sku.lower() in question_lower
    ]

    if mentioned_skus:
        relevant_data = all_flags_df[all_flags_df["sku_id"].isin(mentioned_skus)]
    else:
        relevant_data = all_flags_df[all_flags_df["risk_flag"].apply(lambda x: "ok" not in x)]
        if relevant_data.empty:
            relevant_data = all_flags_df

    data_text = relevant_data.to_string(index=False)

    prompt = (
        "You are answering a business user's question about supply chain risk data. "
        "Use only the data provided below, do not invent numbers or SKUs that are not "
        "listed. If the data does not contain enough information to answer, say so.\n\n"
        f"Relevant data:\n{data_text}\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": "You answer questions about supply chain data clearly and simply, grounded only in the data you are given."},
        {"role": "user", "content": prompt},
    ]
    return call_openrouter(messages)
