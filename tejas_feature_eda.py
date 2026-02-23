import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def main():
    print("Loading base demand data...")
    # Load base demand data
    demand_df = pd.read_csv('demand_monthly.csv')
    
    # 1. Feature Engineering: Brand Pricing Tiers
    # Map brands to predefined pricing/market tiers. 
    # Adjust this mapping based on domain knowledge if needed.
    print("Engineering Brand_Tier feature...")
    brand_tier_map = {
        'Nike': 'Sport',
        'Lacoste': 'Premium',
        'Calvin Klein': 'Premium'
    }
    # Fallback to 'Other' if brand is not in map
    demand_df['Brand_Tier'] = demand_df['BrandName'].map(brand_tier_map).fillna('Other')

    # 2. Feature Engineering: "Lookalike" Product ID
    print("Engineering Lookalike_ID feature...")
    # Fill NAs with strings to ensure concatenation works
    brand_feat = demand_df['BrandName'].fillna('UnknownBrand')
    color_feat = demand_df['Color_Base'].fillna('UnknownColor')
    mat_feat = demand_df['Material'].fillna('UnknownMaterial')
    demand_df['Lookalike_ID'] = brand_feat + "_" + color_feat + "_" + mat_feat

    # Convert 'Month' to datetime for merging with Google Trends
    demand_df['Date'] = pd.to_datetime(demand_df['Month'])

    # 3. Merge original Google Trends
    print("Merging original Google Trends...")
    orig_trends = pd.read_csv('opticalsun_googletrends.csv')
    orig_trends['Date'] = pd.to_datetime(orig_trends['Time'])
    # Drop 'Time' to avoid duplication
    orig_trends = orig_trends.drop(columns=['Time'])
    demand_df = demand_df.merge(orig_trends, on='Date', how='left')

    # 4. Merge New Google Trends (Brands, Materials, FSA)
    print("Merging new Google Trends datasets...")
    trends_path = 'VSP Vision Datasets/google trends/'
    trend_files = glob.glob(os.path.join(trends_path, '*.csv'))
    
    for file in trend_files:
        temp_df = pd.read_csv(file)
        
        # Ensure it has exactly two columns: 'Time' and '{Search_Term}'
        if 'Time' in temp_df.columns and len(temp_df.columns) == 2:
            val_col = temp_df.columns[1] # E.g., 'FSA glasses'
            
            # Rename the column to be a bit more pythonic
            clean_col_name = "Trend_" + val_col.replace(' ', '_')
            temp_df = temp_df.rename(columns={val_col: clean_col_name})
            
            temp_df['Date'] = pd.to_datetime(temp_df['Time'])
            temp_df = temp_df.drop(columns=['Time'])
            
            # Merge into demand
            demand_df = demand_df.merge(temp_df, on='Date', how='left')
        else:
            print(f"Skipping {file} due to unexpected structure.")

    # Sort and clean up
    demand_df = demand_df.sort_values(by=['Region', 'GridValue', 'Date']).reset_index(drop=True)
    
    # Save the enriched dataset
    output_filename = 'demand_monthly_enriched.csv'
    demand_df.to_csv(output_filename, index=False)
    print(f"Successfully saved enriched dataset to {output_filename}")
    print(f"New shape: {demand_df.shape}")

    # --- EDA Visualization ---
    print("\nGenerating EDA Plots...")
    os.makedirs('EDA_Plots', exist_ok=True)
    
    # Target proxy: we don't have strictly '4m_demand' here yet unless we group,
    # so we will plot the distribution of strictly monthly 'Demand' for now, 
    # to understand standard monthly distributions.
    if 'Demand' in demand_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(demand_df['Demand'].dropna(), bins=50, kde=False)
        plt.title('Distribution of Monthly Demand')
        plt.xlabel('Monthly Demand (units)')
        plt.ylabel('Frequency')
        # Limiting x-axis if there are extreme outliers
        plt.xlim(0, demand_df['Demand'].quantile(0.99)) 
        plt.tight_layout()
        plt.savefig('EDA_Plots/demand_distribution.png')
        plt.close()
        print("Saved EDA_Plots/demand_distribution.png")
    
    # Trend vs. Reality
    # Group by Date to get total actual sales per month, compare against General Glasses Trend
    if 'Trend_FSA_glasses' in demand_df.columns:
        monthly_sales = demand_df.groupby('Date')['Demand'].sum().reset_index()
        # Get one trend value per date (since it's replicated across rows)
        monthly_trend = demand_df[['Date', 'Trend_FSA_glasses']].drop_duplicates()
        
        plot_df = monthly_sales.merge(monthly_trend, on='Date', how='inner')
        plot_df = plot_df.sort_values('Date')
        
        # Plot with dual axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Actual Demand', color=color)
        ax1.plot(plot_df['Date'], plot_df['Demand'], color=color, marker='o', label='Actual Demand')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Google Trend: "FSA glasses"', color=color)  
        ax2.plot(plot_df['Date'], plot_df['Trend_FSA_glasses'], color=color, linestyle='--', marker='x', label='FSA Trend')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Actual Eyewear Demand vs. FSA Search Trend over Time')
        plt.savefig('EDA_Plots/trend_vs_reality_fsa.png')
        plt.close()
        print("Saved EDA_Plots/trend_vs_reality_fsa.png")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
