import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def print_top_features(model, feature_names, top_n=10):
    """Utility to print top positive and negative coefficients from a linear model."""
    coefs = model.coef_
    
    # Sort indices by coefficient value
    sorted_idx = np.argsort(coefs)
    
    print(f"\n--- Top {top_n} Negative Drivers (Reduces Sales) ---")
    for idx in sorted_idx[:top_n]:
        print(f"{feature_names[idx]}: {coefs[idx]:.4f}")
        
    print(f"\n--- Top {top_n} Positive Drivers (Increases Sales) ---")
    # Reverse the array for top positive
    for idx in sorted_idx[::-1][:top_n]:
        print(f"{feature_names[idx]}: {coefs[idx]:.4f}")

def main():
    print("Loading final_demand.csv...")
    df = pd.read_csv('final_demand.csv')
    
    # Ensure Date is datetime and sort chronologically for Walk-Forward Validation
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    target_col = '4m_demand'
    
    # Define feature types
    target_encode_cols = ['Lookalike_ID', 'Style']
    one_hot_cols = ['Brand_Tier', 'Shape', 'FrameType', 'Color_Base', 'Color_Finish', 'Material']
    
    # Extract all numeric trend features (including boolean seasons which we treat as numeric 0/1)
    numeric_cols = [col for col in df.columns if col.startswith('Trend_') or col.startswith('is_')]
    
    # Dropping columns that are identifiers or raw targets
    # 'Unnamed: 0', 'Collection', 'BrandLine', 'StyleCode', 'GridValue', 'Region', 'Color', 'BrandName', 'Glasses', 'Sunglasses'
    
    print("\nSetting up Walk-Forward Split (chronological)...")
    split_idx = int(len(df) * 0.8) # 80/20 chronological split
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[target_encode_cols + one_hot_cols + numeric_cols]
    y_train = train_df[target_col].clip(lower=0) # ensure no negatives, required for Poisson
    
    X_test = test_df[target_encode_cols + one_hot_cols + numeric_cols]
    y_test = test_df[target_col].clip(lower=0)
    
    print(f"Training shapes -> X: {X_train.shape}, Y: {y_train.shape}")
    print(f"Testing shapes  -> X: {X_test.shape}, Y: {y_test.shape}")
    
    # --- Preprocessing Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('te', TargetEncoder(target_type='continuous'), target_encode_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_cols)
        ]
    )
    
    # --- Model 1: Ridge Regression ---
    print("\n===============================")
    print("Training Model 1: Ridge Regression")
    print("===============================")
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    
    ridge_pipeline.fit(X_train, y_train)
    y_pred_ridge = ridge_pipeline.predict(X_test)
    
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
    print(f"Ridge Test MAE:  {mae_ridge:.4f}")
    print(f"Ridge Test RMSE: {rmse_ridge:.4f}")
    
    # Extract Feature Names from Pipeline
    cat_enc = ridge_pipeline.named_steps['preprocessor'].named_transformers_['ohe']
    ohe_features = cat_enc.get_feature_names_out(one_hot_cols)
    feature_names = numeric_cols + target_encode_cols + list(ohe_features)
    
    print_top_features(ridge_pipeline.named_steps['model'], feature_names)


    # --- Model 2: Poisson Regression ---
    print("\n===============================")
    print("Training Model 2: Poisson Regression")
    print("===============================")
    poisson_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Using a very slight regularization term
        ('model', PoissonRegressor(alpha=1e-4, max_iter=1000)) 
    ])
    
    poisson_pipeline.fit(X_train, y_train)
    y_pred_poisson = poisson_pipeline.predict(X_test)
    
    mae_poisson = mean_absolute_error(y_test, y_pred_poisson)
    rmse_poisson = root_mean_squared_error(y_test, y_pred_poisson)
    print(f"Poisson Test MAE:  {mae_poisson:.4f}")
    print(f"Poisson Test RMSE: {rmse_poisson:.4f}")
    
    # For Poisson, coefficients are on log-scale but directionality is the same
    print_top_features(poisson_pipeline.named_steps['model'], feature_names)
    
    print("\nBaseline Execution Complete.")

if __name__ == "__main__":
    main()
