# Team5

Eyewear Demand Forecasting

## Background
VSP Vision is a leading provider of eye care services in the U.S., sitting at the intersection of healthcare and retail. Effective supply chain management requires maintaining the right inventory levels. Too much inventory leads to unnecessary costs, while too little results in missed sales opportunities. Demand forecasting is the process of predicting future customer demand for a product or service using historical data, market trends, and other influencing factors (seasonality, promotions, or economic conditions). As an eyewear frame manufacturer,
VSP releases new frames to the marketplace on a periodic basis. These frame releases are called seasons. In preparation for the new season, VSP must order the frames from the manufacturer(s) months in advance. VSP orders inventory equal to 4 months of frame sales as the initial order. The new frame sales eyewear demand forecast is used to generate the initial frame order amount. For this project, the goal is to create an accurate initial new frame sales eyewear demand forecast for all frames for the season. As these are new frames, there is no prior frame sales data for these specific frames to use in the forecast.
The Data Kathleen Lovett and the team at VSP Vision have provided a sample of demand for 9/2023 – 8/2024 in AO-BI275 DEMAND KC KP LA LS KO KS 12.17.25
Calvin Klen_Sept24 ATP, LCAOSTE_Sept24 ATP, and Nike_Sept24 ATP provide product
information for each brand.
The Tasks For this final project, the assignment is to create a new frame sales demand forecast model to predict how much product should be ordered in the initial order using historical frame data
sales, develop a Machine Learning (ML) model to predict frame sales at the Frame
Style/Color/Size level.
Create a ten-minute presentation and a 10-20 page report that answers the following questions.
Since the presentation is relatively short, you may choose to focus on a few highlights from your
analysis and expand further on the details in the report.
1. General analysis:
a. For each frame style size color combination in a season, how many frames should
be ordered?
b. What are the key features that drive frame sales?
c. What could have made your model more accurate?
The intention behind the assignment is to gain a deeper understanding of the supply chain
planning role and how tools are used to analyze their data to create the ideal inventory count to
ensure that we do not have too much or too little product. Creating the data model and
interpreting the outputs, is sufficient to do this project. You are free to do some outside research
to validate/supplement the findings, especially if you are unfamiliar with the domain, and to
apply other more sophisticated text analysis techniques you may know of, but they should not
be required. Large language models are not required for this analysis; if you use them; please
complement the use of LLMs with other techniques (e.g. topic modeling).
Please submit a .pdf of your slides, as well as the final report. There are no hard guidelines for
the final report, but if you are submitting more than 20 pages, you are probably including too
much detail. Similarly, if your report is fewer than 10 pages, you may not be answering the
questions fully. The 10-20 pages is for the text of the report, and does not include the code. The
preferred submission format for the final report is a .pdf file accompanied by the code in a
separate Jupyter notebook. The final report should be separate from your slides and the code
Rubric
Each team should turn in separate files for the presentation, the season 434 scoring
spreadsheet, the report and a Jupyter notebook for the code. The presentation and report will be
graded based on clarity, organization and presentation/writing quality as well as content. Watch
for typos and grammatical mistakes. Structure your presentation and report to tell a coherent
story: with executive summary (one slide/page with key results), introduction/motivation that
states why you are performing the analysis and a conclusion that reinforces key
findings/recommendations.
Presentation: 10 points total, 8 points for answering the questions below, and 2 points for
presentation quality. Focus the presentation on the problem you are solving and any
insights/results, and leave the details to the report.
• What problem is being solved? Focus on one or two key insights from your results, not all
ranges of assortments that the data could address.
• What does the data look like based on your exploratory data analysis?
• How as the data prepared for modeling?
• What modeling approach did you use?
• What are the results?
• How did you evaluate the results?
• What next steps would you recommend based on your results?
Presentations are in class on Feb 25. Please submit separate files for your slides, final
reports and code notebooks.
Only one team member needs to submit the slides and final reports on behalf of the team.
Remember to include a paragraph in your report describing who did what in your team. Report
and code: 10 points total for the report/deliverable, 8 points for answering the questions above, and 2 points for report
- comments.
- VSP
- 
- ---
- 
- ## Execution Instructions & Final Findings
- 
- ### How to Run the Pipeline
- The codebase is structured sequentially from data engineering to final LLM augmentation. Run the scripts in this order:
- 1. **`tejas_feature_eda.py`**: Merges the raw demand, style, and 6 Google Trends datasets. Generates `final_demand.csv`.
- 2. **`tejas_modeling_baseline.py`**: Runs a Ridge Regression (interpretable baseline) using Walk-Forward Validation. (MAE ~158).
- 3. **`tejas_modeling_champion.py`**: Runs a CatBoost model handling categorical data to discover non-linear relationships. Generates `final_order_predictions.csv` (MAE ~113).
- 4. **`tejas_modeling_advanced.py`**: Engineers Time-Series Lags (T-3, T-6), Momentum Deltas, and Sibling Cannibalization Density. Reruns CatBoost to achieve the lowest pure ML error (MAE ~104).
- 5. **`tejas_llm_augmentation.py`**: Requires a `.env` file with `ANTHROPIC_API_KEY`. It runs the Top 5 most vital frame predictions through `claude-opus-4-6` to qualitatively adjust the quantitative baseline.
- 
- ### Key findings & Presentation Answers
- - **A. Optimal Order Quantities:** The exact frame-level order predictions for the next season are generated mathematically in `final_order_predictions.csv`.
- - **B. Key Drivers of Sales:** Intrinsic **Style (33%)** and **Geographic Region (18%)** govern the baseline volume. However, the newly engineered **Sibling Frame Density (15%)** proved that cannibalization (launching too many similar frames at once) heavily depresses individual frame demand. 
- - **C. How to Improve Accuracy (<50 MAE):** Tree-based ML models natively hit a floor around ~104 MAE due to missing internal business context. We proved via Anthropic's `claude-opus-4-6` API that adding qualitative reasoning (knowing that matte black lacks seasonal hype) can further reduce MAE to ~1965 on top-volume outliers. However, true sub-50 MAE requires VSP to provide unconstrained supply chain data: **MSRP (Price), Promotional Calendars, and Historical Stockout Flags**.
- 