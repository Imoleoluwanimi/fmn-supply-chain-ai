# FMN Supply Chain Demand Forecasting & Risk Monitor

## 1. Problem Understanding

The goal of this project is to help Flour Mills of Nigeria (FMN) anticipate potential inventory problems before they happen.

The available data contains daily sales, inventory, replenishment, and supplier lead-time information for multiple SKUs. I interpreted the business problem as two connected tasks:

1. **Forecast future demand** for each SKU using its historical demand patterns.
2. **Translate the forecast into an inventory risk signal** by comparing expected demand with the inventory currently available and the time required to replenish it.

The system therefore separates prediction from business decision-making:

* A machine learning model forecasts expected demand.
* A rule-based risk engine uses the forecast, current stock, demand trend, and lead time to identify potential stockout or overstock risk.
* A Gemini-powered explanation layer translates the result into simple business language for users.

The LLM does not make the inventory-risk decision. It explains a decision that has already been produced by the forecasting and risk-engine components.

---

## 2. Approach

### High-Level Architecture

```text
Raw Supply Chain Data
        |
        v
Data Validation & Cleaning
        |
        v
SKU History Analysis
        |
        +----------------------+
        |                      |
        v                      v
Established SKUs           New SKUs
        |                      |
        v                      v
Feature Engineering       Simple Early Estimate
        |
        v
Demand Forecasting Model
        |
        v
Expected Demand
        |
        v
Inventory Risk Engine
        |
        +-----------------------------+
        |              |              |
        v              v              v
    Stockout       Overstock          OK
      Risk           Risk
        |
        v
Streamlit Dashboard
        |
        v
Gemini Explanation / Q&A
```

### Data

The dataset contains **4,551 observations across 28 SKUs**, covering the period from **January 1, 2026 to June 29, 2026**.

The main fields are:

* `date` - observation date
* `sku_id` - product identifier
* `category` - product category
* `units_sold` - units sold on that day
* `units_received` - units received through replenishment
* `closing_stock` - inventory remaining at the end of the day
* `lead_time_days` - expected replenishment lead time

The dataset contains 25 established SKUs and 3 SKUs with only around 12 days of history.

---

## 3. Data Preparation

The data preparation pipeline:

* validates that the required columns are present
* converts dates into the correct format
* standardizes category values
* checks lead-time consistency for each SKU
* sorts observations by SKU and date
* handles missing demand and inventory values
* separates new SKUs from established SKUs
* creates historical demand features for established SKUs

### New vs Established SKUs

A history threshold of **30 observations** was used to separate new SKUs from established SKUs.

This was necessary because the forecasting features depend on historical observations such as 7-day and 14-day demand patterns.

The three new SKUs did not have enough history to make the same model-based forecast reliable.

Instead, the system uses their average daily sales and latest inventory to produce an early inventory estimate and explicitly labels the result:

> Early estimate, limited history

This prevents the system from presenting a low-history estimate with the same confidence as a model-based forecast.

---

## 4. Demand Forecasting

The forecasting problem is a **time-series forecasting problem formulated as supervised regression**.

Rather than using a traditional statistical forecasting model such as ARIMA, historical demand was transformed into features that a regression model could learn from.

The main features include:

* `sold_lag_1` - demand from the previous day
* `sold_lag_7` - demand from seven days earlier
* `sold_roll_mean_7` - recent 7-day average demand
* `sold_roll_mean_14` - recent 14-day average demand
* `demand_trend` - change between recent and previous demand averages
* `day_of_week` - captures weekly demand patterns

The rolling features are shifted so that the model uses information available before the prediction period rather than the actual demand being predicted.

### Why this approach?

Several approaches could be used for demand forecasting, including:

* moving-average/statistical forecasting
* ARIMA-type models
* feature-based machine learning
* deep learning sequence models such as LSTMs

A 7-day historical average was first used as a baseline.

For the machine-learning approach, Linear Regression, Random Forest and Gradient Boosting were compared using the same temporal train/test setup.

The feature-based approach was selected because it allowed historical demand, recent trends and calendar information to be combined in one model while keeping the solution relatively simple and appropriate for the size of the available dataset.

---

## 5. Model Selection and Evaluation

A chronological train/test split was used rather than randomly shuffling the data.

The model was trained on observations up to **May 30, 2026** and evaluated on the subsequent period from **May 31 to June 29, 2026**.

This preserves the real-world forecasting setup: the model learns from the past and is tested on a later period.

### Model comparison

| Model                  |   MAE |   MAPE |
| ---------------------- | ----: | -----: |
| 7-day average baseline | 63.16 | 24.46% |
| Linear Regression      | 56.59 | 22.46% |
| Random Forest          | 58.42 | 22.47% |
| Gradient Boosting      | 55.59 | 21.56% |

Gradient Boosting produced the lowest error among the candidate models and was selected for the final model.

The model was then tuned using a separate validation window. The selected configuration was:

* `n_estimators = 100`
* `max_depth = 3`
* `learning_rate = 0.05`

### Final performance

On the untouched test period, the final model achieved approximately:

* **MAE: 55.17 units**
* **MAPE: 21.67%**

The model improved on the 7-day-average baseline, which had a test MAE of approximately 63.16 units.

MAE is reported in units because it represents the average absolute difference between predicted and actual demand. MAPE provides a percentage-based view of the forecasting error.

---

## 6. Inventory Risk Engine

The forecast itself does not determine whether an SKU is risky.

The system converts expected demand into a business-oriented inventory metric called **days of cover**:

```text
Days of Cover = Current Stock / Expected Daily Demand
```

This estimates approximately how many days the current inventory can support expected demand.

### Stockout risk

An SKU is flagged as **stockout risk** when:

```text
Days of Cover < Lead Time
```

The reasoning is that the available inventory may be exhausted before replenishment arrives.

### Overstock risk

An SKU is flagged as **overstock risk** when:

```text
Days of Cover > 3 × Lead Time
AND
Demand Trend <= 0
```

This identifies cases where the available inventory covers substantially more time than the replenishment lead time while demand is flat or decreasing.

These thresholds are business heuristics used for the assessment. In a production system, they should be calibrated using historical stockout/overstock outcomes, service-level targets, inventory carrying costs and the business cost of stockouts.

---

## 7. Streamlit Application

The Streamlit application provides an interactive interface for monitoring SKU-level demand and inventory risk.

The dashboard:

* loads the trained forecasting model
* loads precomputed risk results
* displays the total number of SKUs monitored
* highlights SKUs requiring attention
* shows the risk status for individual SKUs
* displays expected demand, current stock, days of cover and lead time
* allows users to upload a new CSV using the expected schema
* reruns the data preparation and risk pipeline on uploaded data
* generates natural-language explanations using Gemini
* supports questions about the available SKU risk data

The application uses the same underlying pipeline for uploaded data rather than relying only on the precomputed results bundled with the project.

---

## 8. Gemini Explanation Layer

Gemini is used as an explanation and interaction layer rather than as the decision-maker.

For each SKU, the application sends the already-computed information, such as:

* SKU
* category
* current stock
* forecasted demand
* days of cover
* lead time
* risk flag
* confidence level

The model is instructed to explain the existing result using only the information provided.

This separation is intentional:

```text
ML Model
   ↓
Forecast
   ↓
Risk Engine
   ↓
Risk Decision
   ↓
Gemini
   ↓
Human-readable Explanation
```

This reduces the risk of an LLM inventing or independently changing an operational risk decision.

The current implementation is better described as **structured-data grounding** rather than traditional vector-database RAG. The application selects relevant rows from the current dataset and provides them directly to the LLM as context.

---

## 9. Project Structure

```text
fmn-supply-chain-ai/
│
├── app/
│   ├── app.py
│   ├── data_prep.py
│   ├── explain.py
│   ├── demand_forecast_model.joblib
│   ├── established_sku_flags.csv
│   ├── new_sku_flags.csv
│   └── requirements.txt
│
└── notebooks/
    ├── 01_data_prep (1).ipynb
    ├── 02_model_and_risk_flags (1).ipynb
    ├── model_comparison.csv
    ├── project1_features.csv
    └── project1_supply_chain_demand (1).csv
```

---

## 10. How to Run

### Requirements

* Python 3.10+
* pip
* Streamlit
* pandas
* numpy
* scikit-learn
* joblib
* requests
* python-dotenv
* Google Gemini API access

Install the Python dependencies with:

```bash
pip install -r app/requirements.txt
```

### Run the Streamlit application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in the browser.

### Gemini API configuration

The Gemini API key should be provided through Streamlit secrets or an environment variable.

Do not hard-code the API key in the source code.

For Streamlit deployment, add the required API key to the application's Secrets configuration.

### Deployed Application

**Streamlit Cloud:**
https://fmn-supply-chain-ai-st7dbdmxbgbxzlspm8ycz7.streamlit.app/

---

## 11. Limitations

This project was developed within a 96-hour technical assessment, so there are several areas that could be improved for production use.

### Limited historical data

The dataset covers approximately six months and only 28 SKUs. More historical data would allow stronger analysis of seasonal patterns and unusual demand periods.

### New SKU forecasting

The three new SKUs have very limited history. The current system therefore uses a simpler early estimate rather than forcing them through the established-SKU forecasting model.

A production system could use category-level information, similar-product behavior or hierarchical/global forecasting to improve cold-start predictions.

### Risk thresholds

The stockout and overstock thresholds are rule-based heuristics. They should be validated against historical inventory outcomes and FMN's actual service-level and inventory-cost requirements.

### Missing-value handling

The current assessment pipeline uses forward and backward filling for some missing values. In a production forecasting system, imputation should be designed carefully around the exact prediction timestamp so that future observations cannot influence past predictions.

### Forecasting approach

Gradient Boosting performed best among the models tested in this assessment, but that does not mean it is universally the best forecasting architecture.

A production implementation should benchmark it against approaches such as ARIMA/exponential smoothing and potentially more advanced global or sequence-based forecasting models as the amount of data grows.

### LLM reliability

Gemini is used only for explanations and data-grounded Q&A, but LLM-generated text can still contain errors. Production deployment would benefit from structured outputs, automated evaluation, stronger guardrails and human review for high-impact decisions.

### Deployment

The current application is designed as an assessment prototype rather than a complete production inventory-management system. A production version would require authentication, monitoring, logging, model/version management, scheduled data pipelines and integration with operational inventory systems.

---

## 12. Next Steps

With additional development time, I would prioritize:

1. Validate risk thresholds with historical FMN inventory outcomes.
2. Add more historical demand data.
3. Benchmark the forecasting model against classical time-series methods.
4. Improve cold-start forecasting for new SKUs.
5. Add automated model retraining as new data becomes available.
6. Add model and data-quality monitoring.
7. Integrate the system with a live inventory or ERP data source.
8. Add authentication and role-based access to the dashboard.
9. Evaluate the LLM explanation layer using a set of known business scenarios.
10. Add alerts for high-priority inventory risks.

---

## 13. Key Takeaway

The project combines **machine learning, business rules and generative AI**, with each component having a clear responsibility:

* **Machine learning** estimates future demand.
* **Business rules** translate demand and inventory information into actionable risk flags.
* **Generative AI** explains those results in language that is easier for a business user to understand.

The key design principle is to keep the operational decision deterministic and use the LLM to improve accessibility and explanation rather than allowing the LLM to make the inventory-risk decision itself.
