import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from spektral.layers import GATConv

from pathlib import Path
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

def build_model(num_features):
    X_in = Input(shape=(num_features,))
    A_in = Input(shape=(None,), sparse=True)

    X = GATConv(
        channels=32,
        attn_heads=2,
        concat_heads=True,
        dropout_rate=0.5,
        activation='elu',
        kernel_regularizer=l2(5e-4)
    )([X_in, A_in])

    X = Dropout(0.5)(X)

    X = GATConv(
        channels=1,
        attn_heads=1,
        concat_heads=False,
        dropout_rate=0.5,
        activation=None,
        kernel_regularizer=l2(5e-4)
    )([X, A_in])

    model = Model(inputs=[X_in, A_in], outputs=X)

    return model

def train():
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading graph data...")
    graph_path = PROCESSED_DIR / 'spektral_graph.pkl'
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found at {graph_path}")

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Convert to TF tensors
    x = tf.convert_to_tensor(graph.x, dtype=tf.float32)
    y = graph.y.astype(np.float32)

    # Convert adjacency to SparseTensor
    coo = graph.a.tocoo()
    indices = np.column_stack([coo.row, coo.col])
    a_sparse = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    a_sparse = tf.sparse.reorder(a_sparse)

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask

    train_labels = y[train_mask]
    num_neg = np.sum(train_labels == 0)
    num_pos = np.sum(train_labels == 1)

    pos_weight = num_neg / (num_pos + 1e-8)

    print(f"Training samples: {len(train_labels)}")
    print(f"Fraud ratio: {num_pos / len(train_labels) * 100:.2f}%")
    print(f"Positive class weight: {pos_weight:.2f}")

    # Per-sample weights: only training nodes get weight > 0
    sample_weights = np.zeros(len(y), dtype=np.float32)
    sample_weights[train_mask & (y == 0)] = 1.0
    sample_weights[train_mask & (y == 1)] = pos_weight
    sample_weights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)

    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    model = build_model(num_features=graph.x.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    epochs = 80
    best_pr_auc = 0.0
    patience = 20
    no_improve = 0

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        # --- Training step ---
        with tf.GradientTape() as tape:
            logits = model([x, a_sparse], training=True)
            # Weighted binary cross-entropy
            bce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_tensor, logits=tf.squeeze(logits)
            )
            loss = tf.reduce_sum(bce * sample_weights) / (tf.reduce_sum(sample_weights) + 1e-8)

        grads = tape.gradient(loss, model.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # --- Validation step ---
        val_logits = model([x, a_sparse], training=False)
        probs = tf.nn.sigmoid(tf.squeeze(val_logits)).numpy()

        val_probs = probs[val_mask]
        val_true = y[val_mask]

        if len(np.unique(val_true)) > 1:
            roc_auc = roc_auc_score(val_true, val_probs)
            pr_auc = average_precision_score(val_true, val_probs)

            improved = ""
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                model.save_weights(str(PROCESSED_DIR / 'gat_model.weights.h5'))
                improved = " * SAVED"
                no_improve = 0
            else:
                no_improve += 1

            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}{improved}")

            if no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break

    print("-" * 70)

    # --- Test evaluation ---
    model.load_weights(str(PROCESSED_DIR / 'gat_model.weights.h5'))
    test_logits = model([x, a_sparse], training=False)
    test_probs_all = tf.nn.sigmoid(tf.squeeze(test_logits)).numpy()

    test_probs = test_probs_all[test_mask]
    test_true = y[test_mask]

    if len(np.unique(test_true)) > 1:
        roc_auc = roc_auc_score(test_true, test_probs)
        pr_auc = average_precision_score(test_true, test_probs)
        print(f"\nTest ROC-AUC: {roc_auc:.4f}")
        print(f"Test PR-AUC:  {pr_auc:.4f}")
    else:
        print("\nWarning: Only one class in test set, cannot compute AUC metrics.")

    print(f"Best Val PR-AUC: {best_pr_auc:.4f}")
    print(f"Model saved to: {PROCESSED_DIR / 'gat_model.weights.h5'}")


if __name__ == '__main__':
    train()