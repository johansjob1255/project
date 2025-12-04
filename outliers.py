import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

FILE_PATH = r"C:\Users\johan\OneDrive - Uppsala universitet\training_data_ht2025.xlsx"
OUTPUT_PATH = r"C:\Users\johan\OneDrive - Uppsala universitet\training_data_capped.xlsx"

def load_data(path):
    try:
        print(f"Loading data from: {path}")
        df = pd.read_excel(path, engine='openpyxl')
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ==========================================
# 2. CAPPING LOGIC
# ==========================================

def cap_outliers_iqr(series):
    """
    Caps outliers using the IQR method.
    Returns the capped series.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap the data (Winsorization)
    return series.clip(lower=lower_bound, upper=upper_bound)

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_capped_boxplots(df, features):
    """FIGURE 1: Box plots for CAPPED features"""
    plt.figure(figsize=(15, 10))
    
    rows = 2
    cols = (len(features) // rows) + 1
    
    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(col, fontweight='bold')
        plt.ylabel("Value")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_capped_violins(df, features):
    """FIGURE 2: Violin plots for CAPPED features"""
    plt.figure(figsize=(15, 10))
    
    rows = 2
    cols = (len(features) // rows) + 1
    
    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.violinplot(y=df[col], color='lightgreen')
        plt.title(col, fontweight='bold')
        plt.ylabel("Value")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_windspeed_comparison(df_original, df_capped):
    """FIGURE 3: Side-by-side comparison for Windspeed"""
    plt.figure(figsize=(12, 6))
    plt.suptitle("Figure 3: Windspeed Capping Effects", fontsize=16)
    
    # Original
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_original['windspeed'], color='salmon')
    plt.title("Original Windspeed (With Outliers)", fontweight='bold')
    plt.ylim(bottom=0)
    
    # Capped
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_capped['windspeed'], color='lightgreen')
    plt.title("Capped Windspeed (Outliers Removed)", fontweight='bold')
    plt.ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Load Data
    df = load_data(FILE_PATH)
    
    if df is not None:
        numerical_cols = ['temp', 'dew', 'humidity', 'windspeed', 'cloudcover']
        valid_cols = [c for c in numerical_cols if c in df.columns]

        # 2. Apply Capping FIRST
        print("Capping outliers...")
        df_capped = df.copy()
        for col in valid_cols:
            df_capped[col] = cap_outliers_iqr(df_capped[col])

        # 3. Generate Figures using CAPPED data
        print("Generating Figure 1 (Box Plots - Capped)...")
        plot_capped_boxplots(df_capped, valid_cols)

        print("Generating Figure 2 (Violin Plots - Capped)...")
        plot_capped_violins(df_capped, valid_cols)

        # 4. Generate Figure 3 (Comparison)
        if 'windspeed' in df.columns:
            print("Generating Figure 3 (Windspeed Comparison)...")
            plot_windspeed_comparison(df, df_capped)
        
        # 5. Save Data
        try:
            df_capped.to_excel(OUTPUT_PATH, index=False)
            print(f"Capped data saved to: {OUTPUT_PATH}")
        except Exception as e:
            print(f"Could not save file: {e}")