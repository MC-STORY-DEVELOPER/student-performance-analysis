# Student Academic Performance Analysis

## Overview

This repository contains an advanced data science project focused on analyzing and predicting student academic performance. Utilizing a comprehensive dataset of factors influencing student success, we employ state-of-the-art machine learning techniques to identify key performance drivers and segment the student population into actionable personas.

The project demonstrates a production-grade data science workflow, including modular code architecture, automated data acquisition, rigorous exploratory data analysis (EDA), predictive modeling with XGBoost, and model interpretability using SHAP (SHapley Additive exPlanations).

## Dataset

The analysis is based on the **Student Performance Factors** dataset, sourced from Kaggle.

*   **Source**: [Student Performance Dataset on Kaggle](https://www.kaggle.com/datasets/ayeshasiddiqa123/student-perfirmance)
*   **Description**: The dataset includes variables such as attendance, hours studied, parental involvement, access to resources, and various other socio-economic factors.
*   **Target Variable**: `Exam_Score`

## Key Features

*   **Automated Data Pipeline**: Scripts to automatically download, validate, and preprocess data using `kagglehub`.
*   **Advanced EDA**: Comprehensive univariate and bivariate analysis to uncover initial correlations and data distributions.
*   **Predictive Modeling**: Implementation of Ensemble methods (XGBoost, Random Forest) with Hyperparameter tuning via RandomizedSearchCV to predict exam scores with high accuracy ($R^2 \approx 0.75$).
*   **Model Interpretability**: Integration of SHAP values to provide global and local explanations for model predictions, offering transparency into *why* a student is predicted to achieve a certain score.
*   **Student Segmentation**: Unsupervised learning (K-Means Clustering) to identify distinct student profiles (e.g., "High Potentials", "At Risk") based on behavioral patterns.

## Repository Structure

```text
.
├── analysis/               # Analysis artifacts
│   ├── plots/              # Generated visualizations (SHAP, Clustering, EDA)
│   ├── student_performance.csv # Local copy of the dataset (downloaded)
│   └── analysis_results.md # Detailed Markdown report of findings
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── loader.py           # Data loading and validation logic
│   ├── preprocess.py       # Scikit-learn pipelines for transformation
│   ├── model.py            # Model training and evaluation
│   ├── analysis.py         # SHAP and Clustering logic
│   └── vis.py              # Visualization utilities
├── main.py                 # Main entry point for the analysis pipeline
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## Installation

### Prerequisites

*   Python 3.8+
*   pip

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/ihuzaifashoukat/student-performance-analysis.git
    cd student-performance-analysis
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To execute the full analysis pipeline, including data download, processing, training, and report generation, run:

```bash
python main.py
```

The script will:
1.  Download the dataset if not present.
2.  Clean and preprocess the data.
3.  Train the XGBoost regressor.
4.  Generate performance metrics (RMSE, MAE, R2).
5.  Save SHAP and clustering visualizations to `analysis/plots/`.
6.  Print a summary of cluster characteristics to the console.

## Results Summary

Our analysis identified **Attendance** and **Hours Studied** as the most critical determinants of academic success.

*   **Model Performance**: The XGBoost model achieved an $R^2$ of 0.75.
*   **Insights**:
    *   Attendance has the strongest positive correlation with exam scores.
    *   Students falling into the "At Risk" cluster (Low Attendance, Low Study Hours) score significantly lower on average (approx. 64.7) compared to the "High Performer" cluster (approx. 69.3).

For a detailed breakdown of findings, refer to [analysis/analysis_results.md](analysis/analysis_results.md).

## Contributing

Contributions are welcome. Please refer to `CONTRIBUTING.md` for guidelines on how to submit improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
