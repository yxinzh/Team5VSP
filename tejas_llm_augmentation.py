import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
import anthropic
from sklearn.metrics import mean_absolute_error

# Load environment variables
load_dotenv()

def main():
    print("Loading datasets...")
    # Load the predictions file
    try:
        preds_df = pd.read_csv('final_order_predictions.csv')
        # Load final_demand to extract the raw trend data for standard context
        features_df = pd.read_csv('final_demand.csv', low_memory=False)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return
        
    # We join them to get the Google Trend features
    # preds_df has: Style, Size, Color_Base, Color_Finish, Region, Date, 4m_demand, Predicted_4m_Order_Quantity
    df = preds_df.merge(features_df, on=['Style', 'Size', 'Color_Base', 'Color_Finish', 'Region', 'Date', '4m_demand'], how='left')
    
    # Pick the top 5 most important predictions (e.g., highest actual demand)
    top_5 = df.nlargest(5, '4m_demand').copy()
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in .env file.")
        return
        
    print("Initializing Anthropic Client...")
    
    # Strictly following user template
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    
    model_name = "claude-opus-4-6"
    print(f"Using Model: {model_name}")
    
    llm_predictions = []
    llm_reasonings = []
    
    import time
    
    for idx, row in top_5.iterrows():
        # Retrieve the specific macro-trends for this item
        fsa_trend = row.get('Trend_FSA_glasses', 0)
        metal_trend = row.get('Trend_metal_frame_glasses', 0)
        
        prompt = f"""You are an expert eyewear supply chain forecaster. 
We are predicting the 4-month initial demand order volume for a specific eyewear frame release.
Our Gradient Boosting model (CatBoost) generated a purely mathematical prediction. Your task is to apply qualitative reasoning to adjust this quantitative baseline.

[FRAME IDENTITY]
- Style Geometry: {row['Style']}
- Base Color: {row['Color_Base']}
- Finish: {row['Color_Finish']}
- Region: {row['Region']}
- Launch Date: {row['Date']}

[MACRO-ECONOMIC CONTEXT (Google Trends Index 0-100)]
- FSA Glasses Search Index: {fsa_trend}
- Metal Frames Search Index: {metal_trend}

[BASELINE FORECAST]
- CatBoost Mathematical Prediction: {row['Predicted_4m_Order_Quantity']} units.

[INSTRUCTIONS]
Current Time-Series LLM research indicates that quantitative models under-predict viral trends and over-predict decaying trends.
Analyze the Frame Identity against the Macro-Economic Context. Does the specific color/style/region combination warrant a percentage adjustment (up or down) from the CatBoost baseline? For example, matte black frames in high-FSA periods might over-index beyond linear math.

You must strictly output ONLY a valid JSON object in the following format, with no markdown formatting or extra text:
{{
    "Adjusted_Prediction": <integer representing new demand volume>,
    "Reasoning": "<A concise explanation of your qualitative adjustment.>"
}}
"""
        print(f"\n--- Processing Style: {row['Style']} ---")
        print(f"Sending API request for CatBoost prediction: {row['Predicted_4m_Order_Quantity']}...")
        
        try:
            start_t = time.time()
            # Strictly following user template
            message = client.messages.create(
                model=model_name,
                max_tokens=2048,
                timeout=15, # Added strict timeout so script does not hang on API unresponsive
                temperature=0.2, # Reduced to 0.2 because we need strictly JSON
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            elapsed = time.time() - start_t
            print(f"API call succeeded in {elapsed:.2f} seconds.")
            
            # Parse response
            result_text = message.content[0].text.strip()
            print(f"Raw API Response: {result_text}")
            
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            result_json = json.loads(result_text)
            
            new_pred = int(result_json.get("Adjusted_Prediction", row['Predicted_4m_Order_Quantity']))
            reason = result_json.get("Reasoning", "No valid reasoning provided.")
            
            llm_predictions.append(new_pred)
            llm_reasonings.append(reason)
            
            print(f"Successfully extracted -> Pred: {new_pred} | Reason: {reason}")
            
        except Exception as e:
            print(f"\n*** FATAL API ERROR for Style {row['Style']} ***")
            print(f"Error Details: {str(e)}")
            llm_predictions.append(row['Predicted_4m_Order_Quantity'])
            llm_reasonings.append(f"API Error: {str(e)}")
            
    top_5['LLM_Predicted_Quantity'] = llm_predictions
    top_5['LLM_Reasoning'] = llm_reasonings
    
    catboost_mae = mean_absolute_error(top_5['4m_demand'], top_5['Predicted_4m_Order_Quantity'])
    llm_mae = mean_absolute_error(top_5['4m_demand'], top_5['LLM_Predicted_Quantity'])
    
    print("\n==================================")
    print(" LLM Augmentation Results (Top 5)")
    print("==================================")
    print(f"CatBoost Baseline MAE (Top 5): {catboost_mae:.2f}")
    print(f"LLM Augmented MAE (Top 5):     {llm_mae:.2f}")
    
    if llm_mae < catboost_mae:
        print(f"-> EXCELLENT: Claude Opus 4.6 improved the MAE by {catboost_mae - llm_mae:.2f} frames!")
    else:
        print(f"-> NO IMPROVEMENT: The LLM did not improve the MAE. CatBoost was better by {llm_mae - catboost_mae:.2f} frames.")
        
    print("\nSample LLM Adjustments:")
    for _, row in top_5.head(3).iterrows():
        print(f"- Frame: {row['Style']} ({row['Color_Base']} {row['Color_Finish']})")
        print(f"  Actual: {row['4m_demand']} | CatBoost: {row['Predicted_4m_Order_Quantity']} | Claude Opus: {row['LLM_Predicted_Quantity']}")
        print(f"  Reasoning: {row['LLM_Reasoning']}\n")
        
    top_5.to_csv('llm_opus_augmented_predictions_top5.csv', index=False)
    print("Saved 'llm_opus_augmented_predictions_top5.csv'.")

if __name__ == "__main__":
    main()
