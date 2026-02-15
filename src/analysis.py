import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import os

def perform_shap_analysis(model, X_train, X_test, feature_names, plots_dir):
    """
    Calculates SHAP values and generates global feature importance plots.
    """
    print("\nComputing SHAP values...")
    
    # Use TreeExplainer for tree-based models
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Assign feature names to shap_values if available
    shap_values.feature_names = feature_names

    # Summary Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_bar.png"))
    plt.close()
    
    # Summary Plot (Dot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_dot.png"))
    plt.close()
    
    print(f"SHAP plots saved to {plots_dir}")

def perform_clustering(X_scaled, n_clusters=3):
    """
    Performs K-Means clustering to identify student personas.
    """
    print("\nPerforming Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score for {n_clusters} clusters: {score:.4f}")
    
    return clusters
