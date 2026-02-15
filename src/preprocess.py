from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def get_preprocessor(df):
    """
    Creates a Scikit-Learn ColumnTransformer for preprocessing.
    Separates numeric and categorical columns locally.
    """
    # Exclude target variable 'Exam_Score' if present in the df passed for schema inference
    X = df.drop(columns=['Exam_Score'], errors='ignore')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric pipeline: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing with 'missing', then OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # easier to track feature names
    )
    
    return preprocessor

def get_feature_names(preprocessor):
    """
    Extracts feature names from the column transformer.
    """
    # This works for sklearn >= 1.0 with verbose_feature_names_out=False
    try:
        return preprocessor.get_feature_names_out()
    except:
        # Fallback or manual extraction if needed
        return None
