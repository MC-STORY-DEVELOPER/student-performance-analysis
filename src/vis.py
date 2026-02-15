import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_clustering_plot(X_pca, clusters, plots_dir):
    """
    Saves a scatter plot of clusters (using PCA for 2D visualization if needed).
    Assumption: X_pca has at least 2 columns.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Student Segmentation (PCA Reduced)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    
    save_path = os.path.join(plots_dir, 'clustering_pca.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Clustering plot saved to {save_path}")

def plot_actual_vs_predicted(y_test, y_pred, plots_dir):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Exam Scores")
    
    save_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
    plt.savefig(save_path)
    plt.close()
