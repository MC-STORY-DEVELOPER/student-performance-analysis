# Student Performance Analysis Report

## 1. Dataset Overview
- **Rows**: 6607
- **Columns**: 20
- **Target Variable**: `Exam_Score`

## 2. Data Quality
- **Missing Values**:
    - `Teacher_Quality`: 78 missing
    - `Parental_Education_Level`: 90 missing
    - `Distance_from_Home`: 67 missing
- **Recommendation**: Imputation or removal of these rows depending on the model requirement. For this EDA, they were kept as is.

## 3. Key Insights (Correlations)
The strongest predictors for `Exam_Score` are:
1.  **Attendance (0.58)**: Strong positive correlation. Regular attendance is highly predictive of better performance.
2.  **Hours_Studied (0.44)**: Significant positive impact. More study hours generally lead to higher scores.
3.  **Previous_Scores (0.17)**: Moderate positive correlation. Past performance is a decent indicator of future results.
4.  **Tutoring_Sessions (0.15)**: Weak positive correlation.
5.  **Physical_Activity** and **Sleep_Hours**: Showing negligible direct correlation with `Exam_Score` in this dataset.

## 4. Visualizations
The following plots have been generated in `analysis/plots`:
- **Distributions**: Histograms for all numeric variables (e.g., `hist_Exam_Score.png`).
- **Counts**: Bar charts for categorical variables (e.g., `count_Parental_Involvement.png`).
- **Correlations**: `correlation_matrix.png` showing the heatmap of numeric features.

## 5. Conclusion
To improve student performance, the data suggests focusing primarily on **Attendance** and **Study Hours**. Interventions that encourage regular school attendance and dedicated study time are likely to yield the best results.
