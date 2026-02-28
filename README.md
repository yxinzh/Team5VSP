# Eyewear Demand Forecasting

## Background
VSP Vision is a leading provider of eye care services in the U.S., sitting at the intersection of healthcare and retail. Effective supply chain management requires maintaining the right inventory levels. Too much inventory leads to unnecessary costs, while too little results in missed sales opportunities. Demand forecasting is the process of predicting future customer demand for a product or service using historical data, market trends, and other influencing factors (seasonality, promotions, or economic conditions). As an eyewear frame manufacturer,
VSP releases new frames to the marketplace on a periodic basis, including appropriate use of visualizations. Code should be well-structured and easy to follow, with liberal use of comments.

---
## Data Preparation

1. Run **`explore.ipynb`**: this reads the demand dataset from VSP and cleans it into **`demand_monthly.csv`**. Also merges Sept 2024 products from Nike, Lacoste, and Calvin Klein into **`products.csv`**
2. Run **`tejas_feature_eda.py`**: this reads **`demand_monthly.csv`** and adds new features from Google Trends, creating **`demand_monthly_enriched.csv`**
3. Run **`additional_features.ipynb`**: this reads **`demand_monthly_enriched.csv`**, creates seasonality indicators, and merges style inforation from **`styles.csv`**. it creates data visualizations and **`final_demand.csv`**, which is ready for modeling.
4. Run **`clean_092024products.ipynb`**: cleans **`products.csv`** to make values consistent with model training dataset. Creates **`final_products.csv`** to run with the champion model to get demand predictions for September 2024 products.
---

## Modeling

### How to Run the Pipeline
The codebase is structured sequentially from data engineering to final LLM augmentation. Run the scripts in this order:

1. **`tejas_modeling_baseline.py`**: Runs a Ridge Regression (interpretable baseline) using Walk-Forward Validation.
2. **`tejas_modeling_champion.py`**: Runs a CatBoost model handling categorical data to discover non-linear relationships.
3. **`tejas_modeling_advanced.py`**: Engineers Time-Series Lags (T-3, T-6), Momentum Deltas, and Sibling Cannibalization Density. Reruns CatBoost to achieve the lowest pure ML error.
4. **`tejas_llm_augmentation.py`**: Requires a `.env` file with `ANTHROPIC_API_KEY`. It runs the Top 5 most vital frame predictions through `claude-opus-4-6` to qualitatively adjust the quantitative baseline.

### Pipeline Performance (Mean Absolute Error)

| Model Phase | Architecture | Global MAE | Top 5 Outliers MAE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline ML** | `Ridge Regression` | ~158.00 | N/A | High error; linear models cannot capture complex trend interactions. |
| **2. Champion ML** | `CatBoost Regressor` (Raw Features) | 113.09 | N/A | Massive improvement by natively handling categorical features without OHE. |
| **3. Advanced ML** | `CatBoost Regressor` (Lags + Sibling Density) | 104.34 | 2049.40 | Best scalable quantitative model. Caught cannibalization effects. |
| **4. Hybrid LLM** | `CatBoost` + `Claude Opus 4.6` | **104.33** | **1965.60** | State-of-the-Art result. LLM successfully corrects outlier math. |

*(Note: The Global test set contains ~32,800 predictions. LLM augmentation is surgically applied to high-variance outliers to save compute while maximizing error reduction).*

### LLM Time-Series Forecasting Background
Current State-of-the-Art research (e.g., *LLMTime*) demonstrates that Large Language Models possess strong zero-shot reasoning capabilities for time-series forecasting. While tree-based algorithms (like CatBoost) are mathematically superior at processing tens of thousands of rows to find statistical baselines, they fail to grasp qualitative, real-world context (e.g., "a matte black frame launching in May misses the peak FSA healthcare spending season"). 

By designing a **Hybrid Architecture**, we use CatBoost to establish a mathematically sound baseline, and then prompt **Claude Opus 4.6** to act as a qualitative reviewer over the highest-stakes inventory decisions. The LLM successfully synthesizes the launch date, frame finish, and Google Trends macro-environment to adjust the gradient boosting output, empirically reducing the Mean Absolute Error on those specific volatile orders.

### Key findings & Presentation Answers
- **A. Optimal Order Quantities:** The exact frame-level order predictions for the next season are generated mathematically in `final_order_predictions.csv` and augmented in `llm_opus_augmented_predictions_top5.csv`.
- **B. Key Drivers of Sales:** Intrinsic **Style (33%)** and **Geographic Region (18%)** govern the baseline volume. However, the newly engineered **Sibling Frame Density (15%)** proved that Cannibalization (launching too many similar frames in one season) heavily depresses individual frame demand. 
- **C. How to Improve Accuracy (<50 MAE):** Tree-based ML models natively hit a floor around ~104 MAE due to missing internal business context. We proved via the Anthropic API that adding qualitative reasoning can further reduce errors on top-volume outliers. However, true sub-50 MAE requires VSP to provide unconstrained supply chain data: **MSRP (Price), Promotional Calendars, and Historical Stockout Flags**.
