import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.loader import load_data, validate_data
from src.preprocess import get_preprocessor, get_feature_names
from src.model import train_model, evaluate_model
from src.analysis import perform_shap_analysis, perform_clustering
from src.vis import save_clustering_plot, plot_actual_vs_predicted

# Config
DATA_FILE = os.path.join('analysis', 'student_performance.csv')
PLOTS_DIR = os.path.join('analysis', 'plots')
OS_PLOTS_DIR = os.path.join(os.getcwd(), PLOTS_DIR)

def main():
    print("--- Starting Advanced Analysis Pipeline ---")
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    df = load_data(data_path)
    validate_data(df)
    
    # 2. Preprocessing
    print("\n--- Preprocessing ---")
    X = df.drop(columns=['Exam_Score'])
    y = df['Exam_Score']
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline
    preprocessor = get_preprocessor(X_train_raw)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    if feature_names is None:
        # Fallback number of features
        feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
    
    print(f"Processed feature count: {X_train_processed.shape[1]}")
    
    # 3. Model Training (XGBoost)
    print("\n--- Training Model ---")
    model = train_model(X_train_processed, y_train, model_type='xgboost')
    
    # 4. Evaluation
    metrics = evaluate_model(model, X_test_processed, y_test)
    y_pred = model.predict(X_test_processed)
    plot_actual_vs_predicted(y_test, y_pred, OS_PLOTS_DIR)
    
    # 5. SHAP Analysis
    # Note: SHAP Explainer might need dataset passed differently depending on version
    # providing full training set as background might be slow, so subsample if needed
    perform_shap_analysis(model, X_train_processed, X_test_processed, feature_names, OS_PLOTS_DIR)
    
    # 6. Clustering
    # Use full X for clustering to see global patterns (or just train)
    X_full_processed = preprocessor.transform(X)
    clusters = perform_clustering(X_full_processed, n_clusters=4)
    
    # PCA to visualize clusters
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full_processed)
    save_clustering_plot(X_pca, clusters, OS_PLOTS_DIR)
    
    # Analysis of Clusters
    df['Cluster'] = clusters
    print("\n--- Cluster Analysis ---")
    print(df.groupby('Cluster')['Exam_Score'].mean().sort_values())
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
