# Eyewear Demand Forecasting: Targeted EDA & Feature Plan

## 1. Top 3 External Data Sources (Google Trends)
Pull these 3 targeted queries from [trends.google.com](https://trends.google.com). 
* **Settings:** United States, Past 5 Years (to cover 9/2023–8/2024 + historical avg), Category: All.
* **Export format:** CSV

| Goal | Exact Search Term | Suggested Filename | Why it moves the needle |
| :--- | :--- | :--- | :--- |
| **Brand Momentum** | `"Nike glasses"` OR `"Lacoste glasses"` OR `"Calvin Klein glasses"` | `trends_brands.csv` | Brand popularity changes over time. A declining brand will sell fewer new frames regardless of the season. |
| **Material Shifts** | `"Metal frame glasses"` OR `"Acetate glasses"` | `trends_materials.csv` | Captures macro fashion shifts between metal and plastic/acetate frames. |
| **FSA Seasonality** | `"FSA glasses"` | `trends_fsa.csv` | FSA funds expire at year-end or mid-March. This quantifies the "insurance reset" effect better than a simple boolean flag. |

---

## 2. Targeted Feature Engineering (High Impact)
On top of your current features, build these 2 to help the model handle the "Cold Start" problem (predicting for brand-new items):

1. **Brand Pricing Tiers:** 
   - Create a categorical feature `Brand_Tier` mapping each brand to `Luxury`, `Premium`, or `Sport`. This groups similar brands so the model can learn broader pricing/demand behavior.
2. **"Lookalike" Product ID:** 
   - Concatenate features: `BrandName_ColorBase_Material` (e.g., `Nike_Black_Plastic`). This creates a micro-segment. When a new frame launches, the model can look at historical demand for this exact "lookalike" segment to estimate baseline sales.

---

## 3. Concise EDA Steps (For the Presentation)
Generate these 2 plots to build a compelling narrative for your final slides:

1. **Target Distribution Check:** Plot a histogram of the `4m_demand` variable. It will likely be heavily right-skewed (few massive sellers, many low sellers). This visualization justifies your choice of ML model (e.g., Poisson regression or using log-transformed demand).
2. **Trend vs. Reality:** Plot a line chart over time showing *Actual Sales* vs *Google Trends Index* (from your original `opticalsun_googletrends.csv`). This visually proves to the graders that external search data is a valid predictor of real-world demand.
