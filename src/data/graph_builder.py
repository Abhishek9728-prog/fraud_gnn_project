import pandas as pd
import numpy as np
import scipy.sparse as sp
from spektral.data import Graph
from pathlib import Path
from tqdm import tqdm
import pickle # to save the model 
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

# Attributes to build edges upon. Sharing these means a connection.
# When should two transactions connect?
# If they share the same card1, card2, DeviceInfo, addr1, P_emaildomain create a edge between them
EDGE_ATTRIBUTES = [
    'card1',
    'card2',
    'addr1',
    'P_emaildomain',
    'R_emaildomain',
    'DeviceInfo',
    'DeviceType',
    'id_12'
]

# Custom initial weights based on attribute importance
ATTRIBUTE_WEIGHTS = {
    'card1': 5.0,
    'card2': 3.0,
    'DeviceInfo': 3.0,
    'DeviceType': 2.0,
    'addr1': 2.0,
    'P_emaildomain': 1.0,
    'R_emaildomain': 1.0,
    'id_12': 2.0
}

MAX_GROUP_SIZE = 50

USE_TEMPORAL_FILTER = True
TIME_WINDOW_SECONDS = 86400
TIME_COLUMN = 'TransactionDT'

def build_graph():
    print("Loading cleaned data...")
    # here now we are loading dataset that we have cleaned and have done preprocess on it .
    df = pd.read_parquet(PROCESSED_DIR / 'cleaned_transactions.parquet') 
    
    # 1. Node Features (x)
    print("Extracting node features...")
    feature_cols = [c for c in df.columns if c != 'isFraud'] # Take all columns except target label
    # so feature colu have all column except isFraud.
    # because isFraud is what we want to predict.
    x = df[feature_cols].values.astype(np.float32) # create the Node Feature Matrix
    # .values convert datframe into numpy matrix
    # .astype(np.float32) convert the numpy matrix into float32
    # Deep learning frameworks expect:tensors,numeric arrays,float computations
    
    # Target (y)
    y = df['isFraud'].values.astype(np.float32) # binary target
    
    # Pre-extract time column to a fast numpy array for temporal filtering
    # Doing df.iloc[i] inside a double loop is extremely slow, so we use a numpy array!
    if USE_TEMPORAL_FILTER and TIME_COLUMN in df.columns:
        time_arr = df[TIME_COLUMN].values
    else:
        time_arr = None

    # 2. Edge Index Construction (Adjacency Matrix)
    print("Constructing edges...")
    # Store edges with accumulated weights
    edge_dict = defaultdict(float) # These both store graph edges. (Combining src/dst into a dict)
    # two becuase Graph libraries often store edges as:(source, destination) pairs.
    
    for attr in EDGE_ATTRIBUTES:# ['card1', 'card2', 'DeviceInfo', 'addr1', 'P_emaildomain']
        if attr not in df.columns:
            continue
        print(f"\nBuilding edges using: {attr}")
        
        # Get the weight for this specific attribute
        weight = ATTRIBUTE_WEIGHTS.get(attr, 1.0)
        
        grouped = df.groupby(attr).groups # Groups rows by the attribute values and returns a dictionary where keys are attribute values and values are lists of row indices. 
        
        for val, indices in tqdm(grouped.items(), desc=attr): # iterate over the dictionary.
            # Skip groups that are too large (memory explosion) or too small (no edges)
            if len(indices) > MAX_GROUP_SIZE or len(indices) < 2:
                # if group is too large we skip it to avoid memory explosion.
                # if group is too small we skip it to avoid no edges.because one node along cannot form edge.
                continue
                
            idx_list = list(indices) # Convert the list of indices to a list.
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)): # create edges between all pairs of nodes in the group.
                    src = idx_list[i]
                    dst = idx_list[j]
                    
                    # Temporal Filtering
                    if time_arr is not None:
                        t1 = time_arr[src]
                        t2 = time_arr[dst]
                        if abs(t1 - t2) > TIME_WINDOW_SECONDS:
                            continue
                    
                    # Accumulate weights. This creates an edge.
                    # extend function -> we used to append to lists, now we accumulate directly into the dictionary
                    edge_dict[(src, dst)] += weight
                    edge_dict[(dst, src)] += weight
                    
    print(f"\nTotal unique edges created: {len(edge_dict)}")
    
    num_nodes = len(df)
    
    # Create SciPy Sparse Adjacency Matrix
    if len(edge_dict) > 0:
        rows = []
        cols = []
        data_list = []

        for (src, dst), w in edge_dict.items():
            rows.append(src)
            cols.append(dst)
            data_list.append(w)

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data_list, dtype=np.float32) # data is edge weight means showing strength of connection

        # sparse matrix contain only non zero entities ,becasue Most nodes are NOT connected.and if we try to show every node it will lead to memory overflow.
        # COO format -> (data, (rows, columns))
        # so it will store only non zero values.
        a = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        #Put value=data[i] at position: (rows[i], cols[i]).

        # Remove self-loops and duplicates because it internally they have self loop and if we add form outside it will make duplicates .
        a.setdiag(0)
        a.eliminate_zeros()
        # Sparse matrices may still store: unnecessary zero entries .This cleans memory.

        # Normalize weights so they are between 0 and 1, which helps Neural Network stability
        max_weight = a.data.max()
        if max_weight > 0:
            a.data = a.data / max_weight

        # Spektral layers usually expect the graph adjacency to have self loops or be normalized,
        # but the layer itself often handles this (like GATConv). 
        # We will keep it raw here and optionally preprocess in the loader/model.
    else:
        a = sp.coo_matrix((num_nodes, num_nodes), dtype=np.float32)
        a = a + sp.eye(a.shape[0])
    # 3. Create Spektral Graph object (store adjacency as CSR for slicing support)
    graph = Graph(x=x, a=a.tocsr(), y=y) # stores node features,adjacency matrix, node labels.
    
    # Generate train/val/test masks (Time-based split: 70/15/15)
    print("Creating masks...")
    train_end = int(num_nodes * 0.7)
    val_end = int(num_nodes * 0.85)
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    
    # Graph Statistics
    print("\n========== GRAPH STATS ==========")
    num_edges = a.nnz
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    sparsity = 1 - (num_edges / (num_nodes * num_nodes)) if num_nodes > 0 else 1
    
    print(f"Nodes           : {num_nodes}")
    print(f"Edges           : {num_edges}")
    print(f"Average Degree  : {avg_degree:.2f}")
    print(f"Sparsity        : {sparsity:.8f}")

    out_path = PROCESSED_DIR / 'spektral_graph.pkl'
    print(f"\nSaving graph to: {out_path}")
    with open(out_path, 'wb') as f:
        pickle.dump(graph, f)
        
    print("\nGraph successfully created!")
    print(graph)

if __name__ == '__main__':
    build_graph()
