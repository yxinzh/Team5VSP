import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def main():
    print("Loading final_demand.csv...")
    # Load and clean
    df = pd.read_csv('final_demand.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Target
    target_col = '4m_demand'
    
    # Feature Selection for CatBoost (Handles categorical natively!)
    categorical_cols = ['Style', 'Region', 'Color_Base', 'Color_Finish', 'Brand_Tier', 'Shape', 'FrameType', 'Material', 'Lookalike_ID']
    numeric_cols = [col for col in df.columns if col.startswith('Trend_') or col.startswith('is_')]
    
    features = categorical_cols + numeric_cols
    
    # Handle any NaNs in categorical columns (CatBoost prefers a string placeholder over NaN)
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('missing')
    
    # Walk-Forward Split (80/20 chronological)
    print("\nSetting up Walk-Forward Split (chronological)...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    print(f"Training shapes -> X: {X_train.shape}, Y: {y_train.shape}")
    print(f"Testing shapes  -> X: {X_test.shape}, Y: {y_test.shape}")

    # Create CatBoost Pools
    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)
    
    print("\n===============================")
    print("Training Champion Model: CatBoost")
    print("===============================")
    
    # Initialize and train CatBoost Regressor
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.08,
        depth=6,
        loss_function='RMSE',
        eval_metric='MAE',
        random_seed=42,
        logging_level='Silent'
    )
    
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred) # Floor predictions at 0 (can't sell negative frames)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print(f"CatBoost Test MAE:  {mae:.4f}")
    print(f"CatBoost Test RMSE: {rmse:.4f}")
    
    # Feature Importances
    print("\n--- Global Feature Importances ---")
    importances = model.get_feature_importance(train_pool)
    feature_names = model.feature_names_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.2f}%")
        
    # Generate Final Output for Rubric A (Frame Style/Size/Color/Demand Mapping)
    # We will score the entire dataset (or just the test set) to show what should be ordered
    print("\nGenerating final order predictions for Report/Presentation...")
    full_pool = Pool(df[features], cat_features=categorical_cols)
    df['Predicted_4m_Order_Quantity'] = np.maximum(0, np.round(model.predict(full_pool)))
    
    # Select columns specifically requested by Rubric Question A
    output_df = df[['Style', 'Size', 'Color_Base', 'Color_Finish', 'Region', 'Date', '4m_demand', 'Predicted_4m_Order_Quantity']]
    output_filename = 'final_order_predictions.csv'
    output_df.to_csv(output_filename, index=False)
    
    print(f"Saved prediction template to {output_filename}")
    print("Execution Complete.")

if __name__ == "__main__":
    main()
