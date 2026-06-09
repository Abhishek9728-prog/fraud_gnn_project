import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

def train_xgboost():
    print("Loading preprocessed tabular data...")
    data_path = PROCESSED_DIR / 'cleaned_transactions.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at {data_path}. Please run preprocess.py first.")
        
    df = pd.read_parquet(data_path)
    
    # Feature columns
    target = 'isFraud'
    feature_cols = [c for c in df.columns if c != target]
    
    X = df[feature_cols].values
    y = df[target].values
    
    num_nodes = len(df)
    train_end = int(num_nodes * 0.7)
    val_end = int(num_nodes * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")
    
    # Calculate pos_weight for XGBoost
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = num_neg / (num_pos + 1e-8)
    print(f"Positive class weight: {pos_weight:.2f}")
    
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        tree_method='hist',
        random_state=42,
        eval_metric='aucpr',
        early_stopping_rounds=30
    )
    
    # Needs a tuple of eval sets
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    print("\nEvaluating on Validation Set...")
    val_preds = model.predict_proba(X_val)[:, 1]
    val_roc = roc_auc_score(y_val, val_preds)
    val_pr = average_precision_score(y_val, val_preds)
    print(f"Validation ROC-AUC: {val_roc:.4f}")
    print(f"Validation PR-AUC:  {val_pr:.4f}")
    
    print("\nEvaluating on Test Set...")
    test_preds = model.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_preds)
    test_pr = average_precision_score(y_test, test_preds)
    print(f"Test ROC-AUC: {test_roc:.4f}")
    print(f"Test PR-AUC:  {test_pr:.4f}")
    
    out_path = PROCESSED_DIR / 'xgb_model.json'
    model.save_model(out_path)
    print(f"\nModel saved to {out_path}")

if __name__ == '__main__':
    train_xgboost()
