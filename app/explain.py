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


def call_gemini(messages, model=DEFAULT_MODEL, max_tokens=2000):
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
    Generate a runtime LLM explanation using the actual computed
    supply chain metrics for the selected SKU.
    """

    risk_flag = str(sku_row.get("risk_flag", "")).lower()

    # Identify the main business issue so the LLM can focus on the
    # actual reason for the flag rather than producing a generic summary.
    if "stockout" in risk_flag:
        risk_driver = "The main concern is that available stock may not last until replenishment arrives."
    elif "overstock" in risk_flag:
        risk_driver = "The main concern is that inventory is high relative to expected demand."
    else:
        risk_driver = "The SKU does not currently show a major inventory risk."

    prompt = f"""
You are explaining a supply chain risk decision to a non-technical business user.

Your job is NOT to make a new prediction. The risk flag has already been
calculated by the application's forecasting and business rules. Your job is
only to explain the decision using the actual numbers provided.

SKU information:
- SKU: {sku_row.get("sku_id")}
- Category: {sku_row.get("category")}
- Current stock: {round(float(sku_row.get("closing_stock", 0)), 2)}
- Forecasted daily demand: {round(float(sku_row.get("forecasted_demand", 0)), 2)}
- Days of cover: {round(float(sku_row.get("days_of_cover", 0)), 2)}
- Lead time: {sku_row.get("lead_time_days")} days
- Risk flag: {sku_row.get("risk_flag")}
- Confidence: {sku_row.get("confidence", "Model-based")}

Main risk driver:
{risk_driver}

Instructions:
1. Explain why THIS particular SKU received THIS particular risk flag.
2. Prioritize the most important risk driver rather than simply listing every number.
3. Use the actual numbers above.
4. Do not invent or calculate numbers that are not provided.
5. Do not change or question the assigned risk flag.
6. Keep the explanation to 2 or 3 natural sentences.
7. Use plain business language and no technical jargon.
8. Do not use a fixed or repetitive template. Vary the sentence structure naturally
   depending on the situation and which metric is most important.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You explain supply chain decisions clearly and naturally. "
                "Your explanations must be grounded entirely in the supplied "
                "SKU data and must not invent facts."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return call_gemini(messages)


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
    return call_gemini(messages)
