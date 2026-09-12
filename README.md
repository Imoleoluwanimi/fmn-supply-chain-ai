# FMN Supply Chain Demand Forecasting & Risk Monitor

A tool that predicts future demand for FMN's products (SKUs) and flags which ones are at risk of running out of stock or sitting with too much stock, then explains the results in plain business language.

Live app: https://fmn-supply-chain-ai-st7dbdmxbgbxzlspm8ycz7.streamlit.app/

## 1. Problem Understanding

The goal here is to help FMN spot inventory problems before they happen. The data has daily sales, stock levels, replenishment, and supplier lead-time information for a number of SKUs.

I broke the problem into two connected steps:
1. Predict future demand for each SKU using its own sales history.
2. Turn that prediction into a risk signal by comparing expected demand against current stock and how long it takes to restock.

The system keeps prediction and decision-making separate:
- A machine learning model predicts expected demand.
- A simple rule-based engine uses that prediction, along with current stock, demand trend, and lead time, to decide if an SKU is at stockout risk or overstock risk.
- A Gemini-powered layer turns the result into plain business language.

The AI does not decide the risk level. It only explains a decision that the forecast and rules have already made.

## 2. Approach

### High-level flow

Raw data goes through cleaning and validation, then SKUs are split into "established" (enough history to forecast) and "new" (too little history). Established SKUs go through feature engineering and a demand forecasting model. New SKUs get a simpler early estimate. The forecast feeds into the inventory risk engine, which flags each SKU as stockout risk, overstock risk, or OK. All of this shows up on a Streamlit dashboard, where Gemini explains the results and answers questions.

### Data

4,551 rows across 28 SKUs, covering January 1 to June 29, 2026. Fields include the date, SKU ID, category, units sold, units received, closing stock, and lead time in days. 25 SKUs have enough history to forecast normally; 3 SKUs only have about 12 days of history.

### Data preparation

The pipeline checks that the right columns are present, fixes date formats, standardizes categories, checks that lead times make sense, sorts everything by SKU and date, fills in missing values, and builds historical demand features (like 7 day and 14 day averages) for the SKUs that have enough history.

### New vs. established SKUs

I used 30 past observations as the cutoff for "enough history." Below that, the forecasting features (like 7 day and 14 day patterns) are not reliable, so the 3 new SKUs instead get an early estimate based on their average daily sales and latest stock, and the app clearly labels this as "Early estimate, limited history" so it is never confused with a real model-based forecast.

### Demand forecasting

This is a time-series problem, but instead of a traditional model like ARIMA, I turned the sales history into features a regression model can learn from: yesterday's demand, demand from a week ago, recent 7 day and 14 day averages, a demand trend value, and the day of the week. All rolling averages are shifted so the model only ever sees information that would genuinely have been available before the day it's predicting.

### Why this approach

Several approaches could work here, from simple moving averages to ARIMA-style statistical models to deep learning sequence models. I compared a 7 day average baseline against Linear Regression, Random Forest, and Gradient Boosting, all tested the same way: trained on data up to May 30, 2026, and tested on the following month, using a chronological split rather than a random one so the setup mirrors real forecasting, learning from the past and being tested on a later period.

| Model | MAE | MAPE |
|---|---|---|
| 7-day average baseline | 63.16 | 24.46% |
| Linear Regression | 56.59 | 22.46% |
| Random Forest | 58.42 | 22.47% |
| Gradient Boosting | 55.59 | 21.56% |

Gradient Boosting had the lowest error and was chosen as the final model (100 trees, max depth of 3, learning rate 0.05). On the untouched test month, it reached an MAE of about 55.17 units and a MAPE of about 21.67%, an improvement over the 63.16 unit baseline. MAE tells you the average size of the error in actual units; MAPE tells you the same thing as a percentage. The feature-based approach was chosen because it combines historical demand, recent trends, and calendar information in one model while staying simple enough for the size of the available data.


### Inventory risk engine

The forecast alone doesn't tell you if an SKU is risky, so I turn it into "days of cover": Days of Cover = Current Stock / Expected Daily Demand


An SKU is flagged **stockout risk** when Days of Cover is less than the lead time, meaning stock could run out before a new order arrives. It's flagged **overstock risk** when Days of Cover is more than 3 times the lead time AND demand is flat or falling, meaning there's a lot more stock on hand than is actually needed soon. These cutoffs are reasonable business rules for this assessment, but in a real deployment they should be tuned using FMN's actual service-level targets and the real cost of stockouts versus excess stock.

### The Streamlit app

Loads the trained model and the pre-computed risk results, shows how many SKUs are being monitored, highlights the ones that need attention, and shows expected demand, current stock, days of cover, and lead time for each SKU. It also lets you upload a new CSV, which reruns the full pipeline (not just the precomputed results) on your new data, generates a plain-language explanation per SKU using Gemini, and answers free text questions about the current SKU data.

### Gemini's role

For each SKU, the app sends Gemini the already-computed numbers (SKU, category, stock, forecast, days of cover, lead time, risk flag, confidence), and Gemini's only job is to explain that result in plain language. It never changes or invents the risk decision. This is closer to what's called structured-data grounding than a full RAG setup with a vector database: the app picks the relevant rows from the current data and hands them straight to Gemini as context.

### Project structure
fmn-supply-chain-ai/
│
├── app/
│ ├── app.py
│ ├── data_prep.py
│ ├── explain.py
│ ├── demand_forecast_model.joblib
│ ├── established_sku_flags.csv
│ ├── new_sku_flags.csv
│ └── requirements.txt
│
└── notebooks/
├── 01_data_prep.ipynb
├── 02_model_and_risk_flags.ipynb
├── model_comparison.csv
├── project1_features.csv
└── project1_supply_chain_demand.csv


## 3. How to Run

### Requirements
- Python 3.10+
- pip
- Streamlit, pandas, numpy, scikit-learn, joblib, requests, python-dotenv
- A Google Gemini API key

Install everything with:
pip install -r app/requirements.txt

### Run the app
From the project root:
streamlit run app/app.py

### Gemini API key
Provide your key through Streamlit secrets or an environment variable. Never hard-code it into the source files. On Streamlit Cloud, add it under the app's Secrets settings.

### Deployed app
Streamlit Cloud: https://fmn-supply-chain-ai-st7dbdmxbgbxzlspm8ycz7.streamlit.app/

## 4. Limitations & Next Steps

This was built inside a 96-hour technical assessment, so there's a lot that would need work before real production use:

- **Limited data:** about six months of history across 28 SKUs. More data would allow a better look at seasonal patterns and unusual demand spikes.
- **New SKU forecasting:** the 3 new SKUs use a simpler early estimate instead of the full model, since they don't have enough history yet. A production version could borrow patterns from similar SKUs or categories to make a smarter cold-start guess.
- **Risk thresholds:** the stockout and overstock cutoffs are reasonable rules, not tuned on FMN's real outcomes. They should be validated against real stockout and overstock history and FMN's actual service-level needs.
- **Missing values:** the current pipeline fills gaps using forward and backward filling. A production system would need to be more careful that future data never leaks into a past prediction.
- **Forecasting model choice:** Gradient Boosting won in this comparison, but that doesn't make it the best choice forever. It should be benchmarked against classical time-series methods and more advanced models as more data comes in.
- **LLM reliability:** Gemini only explains results and answers questions, but LLM output can still be wrong sometimes. A production version would need structured outputs, evaluation, and human review for anything high-impact.
- **Deployment:** this is an assessment prototype, not a production system. Real use would need authentication, monitoring, logging, scheduled data updates, and integration with FMN's actual inventory systems.

With more time, I'd prioritize:
1. Validating the risk thresholds against real FMN inventory outcomes.
2. Adding more historical demand data.
3. Benchmarking against classical time-series forecasting methods.
4. Improving forecasts for new SKUs with limited history.
5. Adding automated retraining as new data comes in.
6. Adding monitoring for model and data quality.
7. Connecting to a live inventory or ERP data source.
8. Adding authentication and role-based access.
9. Testing the LLM explanations against known business scenarios.
10. Adding alerts for high-priority risks.

## 5. Key Takeaway

The project combines machine learning, business rules, and generative AI, with each component having a clear responsibility:
- **Machine learning** estimates future demand.
- **Business rules** translate demand and inventory information into actionable risk flags.
- **Generative AI** explains those results in language that is easier for a business user to understand.

The key design principle is to keep the operational decision deterministic and use the LLM to improve accessibility and explanation, rather than allowing the LLM to make the inventory-risk decision itself.
