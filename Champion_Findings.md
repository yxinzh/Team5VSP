# Champion Modeling Results & Rubric Answers

We executed Walk-Forward Validation on the `final_demand.csv` dataset using **CatBoost Regressor**, a gradient-boosting model that natively handles our complex categorical variables (`Lookalike_ID`, `Brand_Tier`, `Color`, etc.) alongside our macro-economic Google Trends.

**Champion Metrics:**
* **Test MAE:** `113.09`
* **Test RMSE:** `194.32`

This is a massive **~30% reduction in error** compared to our linear Ridge baseline (MAE 158). By capturing non-linear interactions across features, the model is now significantly more accurate for real-world supply chain application.

Here are the direct answers to your presentation rubric questions based on CatBoost:

---

### A. For each frame style size color combination in a season, how many frames should be ordered?
To answer this, our `tejas_modeling_champion.py` script generated a brand new file called **`final_order_predictions.csv`** in your project directory. 
* This file mathematically maps every single combination of `Style`, `Size`, `Color_Base`, `Color_Finish`, and `Region` to the exact recommended order quantity in the `Predicted_4m_Order_Quantity` column.
* You can subset this file for the specific Season 434 and hand it directly over to the supply chain team.

---

### B. What are the key features that drive frame sales?
According to CatBoost's native Global Feature Importance (which measures how much each feature contributes to reducing prediction error across all trees), sales are driven by a hierarchy of intrinsic product identity followed by macro momentum.

**Top 5 Drivers of Demand:**
1. **Style (33.04%)**: The exact parent style geometry of the frame is the undisputed largest differentiator in demand.
2. **Region (18.39%)**: Geographic location heavily dictates volume (e.g., California moving significantly more sunglasses than colder states).
3. **Lookalike_ID (15.01%)**: Our engineered micro-segment (`Brand_Color_Material`) successfully anchors new frames to the historical momentum of past, similar releases.
4. **Color_Base (8.10%)**: Standardized colors (Black, Tortoise, Crystal) matter more than abstract vendor names.
5. **Brand_Tier (6.75%)**: A brand's categorization (Premium vs. Sport) heavily segments its expected volume.

*Note on Google Trends:* While trends like `Trend_Nike_frames` (2.54%) and `Trend_FSA_glasses` (1.82%) appear lower on the global scale, remember that they are time-based features overlapping with intrinsic properties. When a *Nike* frame is released, the Nike trend acts as a critical time-based multiplexer for that specific release.

---

### C. What could have made your model more accurate?
While CatBoost achieved strong results, the supply chain forecast could be further improved by:
1. **Adding True Lags (ARIMA/LSTM integration):** Currently, the model uses Google Trends from the *exact month* the 4-month demand window begins. Supply chains operate on a 6-month delay. We need to build features that look at Google Trends from $T-6$ months to improve true forecasting integrity.
2. **Pricing & Promotions Data:** We are entirely missing MSRP (price) and explicit promotion calendars. A major factor driving a spike in demand isn't just a Google Trend—it's a 30%-off sale. Integrating promotional flag data would dramatically improve model accuracy.
3. **Cannibalization Metrics:** If VSP launches 10 new styles of "Black Nike Square Frames" in the same season, they will cannibalize each other's sales. Our model treats each frame independently. Building a "sibling frame" density feature would prevent the model from over-forecasting identical styles released simultaneously.
