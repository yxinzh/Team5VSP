import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """Augment the dataset with Lags, Momentum Deltas, and Sibling Density."""
    print("Creating advanced features...")
    df = df.copy()
    
    # Ensure sorted chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['GridValue', 'Date']).reset_index(drop=True)
    
    # 1. Sibling Frame Density (Cannibalization)
    # How many frames of the exact same Lookalike_ID launched in this exact month?
    print(" -> Engineering Sibling_Frame_Density...")
    density = df.groupby(['Date', 'Lookalike_ID']).size().reset_index(name='Sibling_Frame_Density')
    df = df.merge(density, on=['Date', 'Lookalike_ID'], how='left')
    
    # 2. Time-Series Google Trend Lags & Momentum
    # Identify trend columns
    trend_cols = [c for c in df.columns if c.startswith('Trend_') or c in ['Glasses', 'Sunglasses']]
    
    print(" -> Engineering Rolling Averages & Lags...")
    # Because there are multiple identical dates (many frames sold in one month),
    # we first create a unique date-trend dataframe to calculate lags cleanly.
    unique_dates = df[['Date'] + trend_cols].drop_duplicates().sort_values('Date').reset_index(drop=True)
    
    for col in trend_cols:
        # Lag by 3 months and 6 months
        unique_dates[f'{col}_lag3'] = unique_dates[col].shift(3)
        unique_dates[f'{col}_lag6'] = unique_dates[col].shift(6)
        
        # Momentum Deltas: (Current / Past) - 1. Safe divide to avoid Inf/NaN.
        # Tells us if the trend is accelerating or decelerating leading up to launch.
        unique_dates[f'{col}_momentum_3m'] = np.where(unique_dates[f'{col}_lag3'] == 0, 0, 
                                            (unique_dates[col] - unique_dates[f'{col}_lag3']) / (unique_dates[f'{col}_lag3'] + 1e-9))
        
        unique_dates[f'{col}_momentum_6m'] = np.where(unique_dates[f'{col}_lag6'] == 0, 0, 
                                            (unique_dates[col] - unique_dates[f'{col}_lag6']) / (unique_dates[f'{col}_lag6'] + 1e-9))
        
        # Fill absolute NAs created by shifting with 0
        unique_dates.fillna(0, inplace=True)
        
    # Merge engineered trend features back into the main forecasting dataframe
    new_trend_feats = [c for c in unique_dates.columns if c != 'Date' and c not in trend_cols]
    df = df.merge(unique_dates[['Date'] + new_trend_feats], on='Date', how='left')
    
    return df

def main():
    print("Loading final_demand.csv...")
    df = pd.read_csv('final_demand.csv', low_memory=False)
    
    # Augment Dataset
    df = create_advanced_features(df)
    
    # Re-sort for Walk-Forward
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    target_col = '4m_demand'
    categorical_cols = ['Style', 'Region', 'Color_Base', 'Color_Finish', 'Brand_Tier', 'Shape', 'FrameType', 'Material', 'Lookalike_ID']
    
    # Remove any potential duplicate column names created by merging
    df = df.loc[:, ~df.columns.duplicated()]
    
    # ALL numeric features: original trends + new lags + new momentous + is_ booleans + density
    numeric_cols = [col for col in df.columns if col.startswith('Trend_') or col.startswith('is_') or col in ['Glasses', 'Sunglasses']]
    momentum_cols = [col for col in df.columns if '_lag' in col or '_momentum' in col or col == 'Sibling_Frame_Density']
    
    # Ensure features list is entirely unique
    features = list(set(categorical_cols + numeric_cols + momentum_cols))
    
    # Clean categoricals
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('missing')
        
    # Walk-Forward Split (80/20)
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

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)
    
    print("\n============================================")
    print("Training Advanced Champion Model: CatBoost")
    print("============================================")
    
    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.08,
        depth=6,
        loss_function='RMSE',
        eval_metric='MAE',
        random_seed=42,
        logging_level='Silent',
        od_type='Iter',
        od_wait=50 # Early stopping to prevent overfitting
    )
    
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred) # Floor at 0
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print(f"\nAdvanced CatBoost Test MAE:  {mae:.4f}  (Previous Baseline: 113.09)")
    print(f"Advanced CatBoost Test RMSE: {rmse:.4f}")
    
    print("\n--- Top 15 Advanced Feature Importances ---")
    importances = model.get_feature_importance(train_pool)
    feature_names = model.feature_names_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    for idx, row in importance_df.head(15).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.2f}%")
        
    print("\nExecution Complete.")

if __name__ == "__main__":
    main()
