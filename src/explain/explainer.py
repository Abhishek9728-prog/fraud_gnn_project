import os

import tensorflow as tf
import numpy as np

def explain_transaction(model, x, a, node_id):
    """
    Generates a gradient-based Saliency Map explanation for why a specific 
    transaction was classified as fraud or clean using TensorFlow.
    Returns a dictionary mimicking PyTorch Geometric's explanation format.
    """
    print(f"Generating TF Saliency explanation for node {node_id}...")
    
    # We need to watch the adjacency matrix. 
    # Spektral layers generally take SparseTensors. To get gradients w.r.t edges,
    # it's easiest if the adjacency is a SparseTensor.
    if not isinstance(a, tf.SparseTensor):
        # Convert SciPy sparse matrix to tf.SparseTensor
        coo = a.tocoo()
        indices = np.column_stack([coo.row, coo.col])
        a = tf.SparseTensor(indices, coo.data, coo.shape)
        a = tf.sparse.reorder(a)
    
    # We will compute gradient of the target node's logit w.r.t the sparse values
    with tf.GradientTape() as tape:
        tape.watch(a.values)
        
        # Forward pass
        logits = model([x, a], training=False)
        target_logit = logits[node_id, 0]
        
    # Compute gradients
    grads = tape.gradient(target_logit, a.values)
    
    if grads is None:
        # Fallback if gradients don't flow (e.g. detached operations)
        edge_mask = np.ones(a.values.shape)
    else:
        # We take the absolute value of the gradients as the edge importance score
        edge_mask = np.abs(grads.numpy())
        # Normalize
        if np.max(edge_mask) > 0:
            edge_mask = edge_mask / np.max(edge_mask)
            
    # Mock PyG explanation object for compatibility with visuals
    class Explanation:
        def __init__(self, edge_mask):
            self.edge_mask = edge_mask
            
    return Explanation(edge_mask)
