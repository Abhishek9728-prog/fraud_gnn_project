import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import pickle
from spektral.layers import GATConv
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import scipy.sparse as sp

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

def build_gnn_model(num_features):
    X_in = Input(shape=(num_features,))
    A_in = Input(shape=(None,), sparse=True)

    X = GATConv(channels=32, attn_heads=2, concat_heads=True,
                dropout_rate=0.5, activation='elu', kernel_regularizer=l2(5e-4))([X_in, A_in])
    X = Dropout(0.5)(X)
    X = GATConv(channels=1, attn_heads=1, concat_heads=False,
                dropout_rate=0.5, activation=None, kernel_regularizer=l2(5e-4))([X, A_in])

    model = Model(inputs=[X_in, A_in], outputs=X)
    return model

def run_ensemble():
    print("Loading datasets...")
    # 1. Load tabular data for XGBoost
    df = pd.read_parquet(PROCESSED_DIR / 'cleaned_transactions.parquet')
    feature_cols = [c for c in df.columns if c != 'isFraud']
    X_tab = df[feature_cols].values
    y = df['isFraud'].values
    
    # 2. Load Graph data for GNN
    with open(PROCESSED_DIR / 'spektral_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
        
    x_gnn = tf.convert_to_tensor(graph.x, dtype=tf.float32)
    coo = graph.a.tocoo()
    indices = np.column_stack([coo.row, coo.col])
    a_sparse = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    a_sparse = tf.sparse.reorder(a_sparse)
    
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    
    # --- XGBOOST PREDICTIONS ---
    print("Loading XGBoost Model and predicting...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(PROCESSED_DIR / 'xgb_model.json')
    xgb_probs_all = xgb_model.predict(xgb.DMatrix(X_tab))
    
    xgb_val = xgb_probs_all[val_mask]
    xgb_test = xgb_probs_all[test_mask]
    
    # --- GNN PREDICTIONS ---
    print("Loading GNN Model and predicting...")
    gnn_model = build_gnn_model(num_features=graph.x.shape[1])
    gnn_model.load_weights(str(PROCESSED_DIR / 'gat_model.weights.h5'))
    gnn_logits = gnn_model([x_gnn, a_sparse], training=False)
    gnn_probs_all = tf.nn.sigmoid(tf.squeeze(gnn_logits)).numpy()
    
    gnn_val = gnn_probs_all[val_mask]
    gnn_test = gnn_probs_all[test_mask]
    
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # --- FIND OPTIMAL BLEND ON VAL SET ---
    print("\nFinding optimal ensemble weights on Validation Set...")
    best_pr = 0
    best_alpha = 0.5
    
    for alpha in np.linspace(0, 1, 101): # alpha is weight for XGBoost
        blend_val = alpha * xgb_val + (1 - alpha) * gnn_val
        pr = average_precision_score(y_val, blend_val)
        if pr > best_pr:
            best_pr = pr
            best_alpha = alpha
            
    print(f"Optimal Weight found! -> {best_alpha:.2f} XGBoost + {1-best_alpha:.2f} GNN")
    print(f"Validation PR-AUC (XGB Alone): {average_precision_score(y_val, xgb_val):.4f}")
    print(f"Validation PR-AUC (GNN Alone): {average_precision_score(y_val, gnn_val):.4f}")
    print(f"Validation PR-AUC (Ensemble):  {best_pr:.4f}")
    
    # --- EVALUATE ON TEST SET ---
    print("\nEvaluating Ensemble on Test Set...")
    blend_test = best_alpha * xgb_test + (1 - best_alpha) * gnn_test
    
    test_roc_xgb = roc_auc_score(y_test, xgb_test)
    test_pr_xgb = average_precision_score(y_test, xgb_test)
    
    test_roc_gnn = roc_auc_score(y_test, gnn_test)
    test_pr_gnn = average_precision_score(y_test, gnn_test)
    
    test_roc_blend = roc_auc_score(y_test, blend_test)
    test_pr_blend = average_precision_score(y_test, blend_test)
    
    print("-" * 50)
    print(f"{'Model':<15} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 50)
    print(f"{'XGBoost':<15} | {test_roc_xgb:.4f}     | {test_pr_xgb:.4f}")
    print(f"{'GNN':<15} | {test_roc_gnn:.4f}     | {test_pr_gnn:.4f}")
    print(f"{'Ensemble':<15} | {test_roc_blend:.4f}     | {test_pr_blend:.4f}")
    print("-" * 50)
    
    # Save the alpha
    with open(PROCESSED_DIR / 'ensemble_weights.pkl', 'wb') as f:
        pickle.dump({'xgb_weight': best_alpha, 'gnn_weight': 1 - best_alpha}, f)
    
    print(f"Saved ensemble weights to {PROCESSED_DIR / 'ensemble_weights.pkl'}")

if __name__ == '__main__':
    run_ensemble()
