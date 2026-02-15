from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def train_model(X, y, model_type='xgboost'):
    """
    Trains a model (XGBoost or Random Forest) with basic hyperparameter tuning.
    """
    if model_type == 'xgboost':
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        }
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'rf'.")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training {model_type} model with RandomizedSearchCV...")
    search.fit(X, y)
    print(f"Best params: {search.best_params_}")
    
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns metrics.
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("\n--- Model Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics
