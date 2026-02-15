import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set(style="whitegrid")

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'student_performance.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def basic_inspection(df):
    print("\n--- Basic Inspection ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Describe ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

def univariate_analysis(df):
    print("\n--- Univariate Analysis ---")
    # Plot histograms for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'hist_{col}.png'))
        plt.close()
        print(f"Saved histogram for {col}")

    # Plot count plots for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'count_{col}.png'))
        plt.close()
        print(f"Saved count plot for {col}")

def bivariate_analysis(df):
    print("\n--- Bivariate Analysis ---")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix.png'))
    plt.close()
    print("Saved correlation matrix")

    # Pairplot (subset due to potential size)
    # create a pairplot for top 5 correlated features with the target variable if known, 
    # but for now, let's just do a pairplot of the first few numeric columns
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols[:5]])
        plt.savefig(os.path.join(PLOTS_DIR, 'pairplot_subset.png'))
        plt.close()
        print("Saved pairplot subset")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        basic_inspection(df)
        univariate_analysis(df)
        bivariate_analysis(df)
        print("\nEDA completed. Plots saved to 'analysis/plots'.")
