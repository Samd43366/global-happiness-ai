"""
Model Training, Cross-Validation Benchmarking, and Serialization Module.

Benchmarks 4 candidate model families (Baseline, RidgeCV, RandomForest, XGBoost)
under Repeated 5-Fold Cross Validation and exports the optimal pipeline artifact.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.features import get_feature_splits, build_preprocessor


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate model families across linear, bagging, and boosting paradigms."""
    models = {
        'Baseline (Mean)': DummyRegressor(strategy='mean'),
        'Ridge Regression': RidgeCV(alphas=np.logspace(-3, 3, 50)),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost (Optimized)': XGBRegressor(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def benchmark_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, Any]]:
    """
    Execute full benchmarking suite across candidate algorithms.
    
    Returns:
        Tuple: (Benchmarking Results DataFrame, Best Pipeline Artifact, Test Metrics Dict)
    """
    X, y = get_feature_splits(df)
    
    # Stratified split by quintiles of target variable to ensure balanced distribution
    y_binned = pd.qcut(y, q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_binned
    )
    
    preprocessor = build_preprocessor()
    models = get_candidate_models()
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    
    results = []
    trained_pipelines = {}
    
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }
    
    best_pipeline = None
    best_r2 = -float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Fit on full training set
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        # Predict on held-out test set
        y_pred = pipe.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        cv_mae_mean = -cv_scores['test_mae'].mean()
        cv_mae_std = cv_scores['test_mae'].std()
        cv_rmse_mean = -cv_scores['test_rmse'].mean()
        cv_rmse_std = cv_scores['test_rmse'].std()
        cv_r2_mean = cv_scores['test_r2'].mean()
        cv_r2_std = cv_scores['test_r2'].std()
        
        results.append({
            'Model Family': name,
            'CV MAE': f"{cv_mae_mean:.3f} ± {cv_mae_std:.3f}",
            'CV RMSE': f"{cv_rmse_mean:.3f} ± {cv_rmse_std:.3f}",
            'CV R²': f"{cv_r2_mean:.3f} ± {cv_r2_std:.3f}",
            'Test MAE': round(test_mae, 3),
            'Test RMSE': round(test_rmse, 3),
            'Test R²': round(test_r2, 3)
        })
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_pipeline = pipe
            best_model_name = name
            
    results_df = pd.DataFrame(results)
    
    # Serialize optimal model artifact
    models_dir = 'models' if os.path.exists('models') else '../models'
    os.makedirs(models_dir, exist_ok=True)
    export_path = os.path.join(models_dir, 'final_model.pkl')
    joblib.dump(best_pipeline, export_path)
    print(f"Optimal model artifact ({best_model_name}) serialized to {export_path}")
    
    test_metrics = {
        'best_model': best_model_name,
        'best_r2': best_r2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'trained_pipelines': trained_pipelines
    }
    
    return results_df, best_pipeline, test_metrics


if __name__ == '__main__':
    from src.data_loader import load_cleaned_data
    df = load_cleaned_data('cleaned_data.csv')
    res_df, best_pipe, info = benchmark_models(df)
    print("\n=== BENCHMARKING RESULTS ===")
    print(res_df.to_string(index=False))
