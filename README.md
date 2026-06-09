# AI-Based Fraud Detection System using Graph Neural Networks

This project implements an advanced fraud detection system using **Graph Attention Networks (GAT)** built on **TensorFlow + Spektral**. It leverages the IEEE-CIS Fraud Detection Dataset to model relationships between transactions, detecting fraud rings through shared attributes like devices, cards, and email domains.

## Features
- **Data Preprocessing**: Loads and cleans the raw CSV, encodes categoricals, scales numerics, and saves to Parquet.
- **Graph Construction**: Connects transactions as nodes if they share identifying attributes (card number, device, email, etc.) within a 24-hour window.
- **Graph Neural Network**: A 2-layer GAT (Graph Attention Network) trained with Focal Loss to handle the highly imbalanced fraud dataset.
- **Explainable AI**: Gradient-based Saliency Maps highlight which edges (connections) most influenced the fraud classification.
- **Streamlit Dashboard**: Interactive visual interface to investigate any transaction, see its neighborhood graph, and understand why it was flagged.

## Tech Stack
- **Model**: TensorFlow 2.x + Spektral (GATConv)
- **Graph Storage**: SciPy Sparse Matrices + Spektral Graph objects
- **Frontend**: Streamlit + Pyvis (interactive graph visualization)
- **XAI**: TensorFlow GradientTape saliency

## Folder Structure
```
fraud_gnn_project/
├── app/
│   ├── app.py          ← Streamlit dashboard
│   └── visuals.py      ← Pyvis subgraph renderer
├── data/
│   └── processed/      ← Preprocessed parquet + graph pkl + model weights
├── src/
│   ├── data/
│   │   ├── preprocess.py    ← Data cleaning & feature engineering
│   │   └── graph_builder.py ← Edge construction & graph creation
│   ├── models/
│   │   ├── gat.py           ← GAT model definition (Spektral-based)
│   │   └── train.py         ← Training loop with Focal Loss & early stopping
│   ├── explain/
│   │   └── explainer.py     ← Gradient saliency XAI
│   └── utils/
│       └── metrics.py       ← ROC-AUC, PR-AUC, F1 metrics
└── requirements.txt
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Place the raw `final_fraud_dataset.csv` in the `../archive/` directory (one level above the project root), then run preprocessing:
```bash
python -m src.data.preprocess
```

### 3. Build the Transaction Graph
```bash
python -m src.data.graph_builder
```

### 4. Train the Model
```bash
python -m src.models.train
```
The best model weights are saved to `data/processed/gat_weights.pkl`.

### 5. Launch the Dashboard
```bash
streamlit run app/app.py
```

> **Note**: Steps 2–4 can be skipped if `data/processed/spektral_graph.pkl` and `data/processed/gat_weights.pkl` already exist.
