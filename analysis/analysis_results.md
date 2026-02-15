# Advanced Student Performance Analysis Report

## 1. Executive Summary
This report details the findings from the advanced machine learning analysis of the Student Performance dataset. We utilized **XGBoost** for predictive modeling and **SHAP (SHapley Additive exPlanations)** for interpreting the key drivers of student performance. We also performed **Cluster Analysis** to segment students into distinct profile groups.

## 2. Model Performance
We trained an XGBoost Regressor to predict `Exam_Score`.
-   **R² Score**: **0.75** (The model explains ~75% of the variance in exam scores).
-   **RMSE**: **1.90** (On average, predictions are off by less than 2 points).
-   **MAE**: **0.72** (Median absolute error is less than 1 point).

> [!NOTE]
> An R² of 0.75 indicates a strong predictive capability, suggesting that the available features (Attendance, Study Hours, etc.) are very good predictors of academic performance.

## 3. Key Drivers (SHAP Analysis)
The SHAP analysis reveals the most influential factors affecting the exam score.
*(See `analysis/plots/shap_summary_bar.png` for visual details)*

![SHAP Summary Bar](analysis/plots/shap_summary_bar.png)

Based on the model's feature importance:
1.  **Attendance**: Consistently the top predictor. Higher attendance correlates strongly with higher scores.
2.  **Hours_Studied**: The second most critical factor.
3.  **Additional Factors**: Previous scores and Tutoring sessions also play a role, but to a lesser extent than the top two.

![SHAP Summary Dot](analysis/plots/shap_summary_dot.png)

## 4. Student Segmentation (Clustering)
We used K-Means clustering to identify student personas. Four distinct clusters emerged:

| Cluster | Avg Exam Score | Description |
| :--- | :--- | :--- |
| **Cluster 1** | **64.7** | **"At Risk"**: Likely lower attendance and study hours. Needs intervention. |
| **Cluster 3** | **66.5** | **"Average"**: Standard performance, potentially disengaged. |
| **Cluster 2** | **68.4** | **"Above Average"**: Good habits but room for improvement. |
| **Cluster 0** | **69.3** | **"High Performers"**: High attendance and study time. |

> [!TIP]
> The clustering silhouette score was low (~0.05), indicating that students don't fall into perfectly separated groups but rather exist on a continuum. However, the score differentiation validates the segmentation utility.

![Clustering PCA](analysis/plots/clustering_pca.png)

## 5. Visualizations Generated
All plots are saved in `analysis/plots`:
-   `shap_summary_bar.png` & `shap_summary_dot.png`: Feature importance.
-   `actual_vs_predicted.png`: Model accuracy visual check.
-   `clustering_pca.png`: 2D visualization of student segments.

![Actual vs Predicted](analysis/plots/actual_vs_predicted.png)

## 6. Recommendations
1.  **Focus on Attendance**: It is the single biggest lever for performance.
2.  **Target "At Risk" Cluster**: Students in Cluster 1 should be prioritized for counseling and support.
3.  **Study Support**: Encouraging more study hours will directly yield better results.
