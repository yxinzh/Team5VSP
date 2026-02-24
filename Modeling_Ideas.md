# ML Modeling Ideas on Enriched Dataset

Now that `demand_monthly_enriched.csv` contains intrinsic product features (`Lookalike_ID`, `Brand_Tier`, colors, sizes) and macro-economic timeseries features (Google Trends), here is how you can structure your modeling approach.

## 1. The Core Modeling Problem
**Type:** Regression predicting a continuous/count variable (`Demand`).
**Nature:** "Cold Start." Since new frames have no historical sales, you CANNOT use standard autoregressive forecasting models like ARIMA or Prophet, because those rely on a product's past sales to predict its future sales.

Instead, we must treat this as a **Supervised Regression Mapping**:
$$f(\text{Product Features}, \text{Time}, \text{Macro Trends}) = \text{Demand}$$

---

## 2. Model Architectures to Leverage

### A. Random Forest / XGBoost (The Strongest Baseline)
Tree-based models are extremely effective for tabular data with mixed categorical and continuous features.
* **Why it fits:** It handles non-linear relationships well (e.g., if "Bright Pink" sells well in Summer but terribly in Winter, a tree easily captures this interaction).
* **Target Variable Setup:** Since we need the *initial 4-month order amount*, you have two choices:
  1. *Direct Approach:* Aggregate the `demand_monthly_enriched.csv` to one row per `GridValue`. The target `Y` is the sum of the first 4 months. The features `X` are the product features and the average Google Trends over those 4 months.
  2. *Monthly Approach:* Train the RF to predict *monthly* demand. To get the final ordering amount, predict month 1, 2, 3, and 4 for the new frame and sum them up.

### B. Recurrent Neural Networks: LSTM (Experimental)
LSTMs are designed to handle sequence data. Because our Google Trends are inherently sequential, an LSTM can capture the "momentum" of a trend leading up to a launch date.
* **Why it fits:** If the search trend for "Metal Frames" has been accelerating over the past 3 months, an LSTM will pick up on this trajectory better than a static Random Forest.
* **Architecture Design:**
  * **Input 1 (Sequential):** A 12-month historical sequence of the Google Trends features (e.g., `Brand_Trend`, `FSA_Trend`) leading up to the frame's launch month. Processed through the LSTM layers.
  * **Input 2 (Static):** The intrinsic frame features (one-hot encoded `Brand_Tier`, `Material`, `Color`). Processed through standard Dense layers.
  * **Fusion:** Concatenate the LSTM output with the Dense output to predict the 4-month demand volume.

---

## 3. Preparation Steps Before Modeling
1. **Target Rollup:** You must decide whether your `Y` is monthly demand or aggregated 4-month demand, and reshape the dataset accordingly.
2. **Encoding Vectors:** Neural networks (like LSTM or simple MLPs) require scaling numerical inputs (Trends) via StandardScaler and embedding or one-hot encoding variables like the `Lookalike_ID`.
3. **Validation Strategy:** Do NOT use standard cross-validation. Since this is time-series, use **Time-Series Split** (walk-forward validation) where you train on months 1-8 and predict on months 9-12 to prevent data leakage from the future.
