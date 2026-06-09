import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
import pickle
import streamlit.components.v1 as components
from pathlib import Path
import sys
import scipy.sparse as sp
import random

from spektral.layers import GATConv
import xgboost as xgb
from spektral.data import Graph

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'src'))
sys.path.insert(0, str(ROOT_DIR / 'app'))

from src.explain.explainer import explain_transaction
from visuals import render_subgraph


def build_model(num_features):

    X_in = Input(shape=(num_features,))
    A_in = Input((None,), sparse=True)

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


st.set_page_config(
    page_title="GNN Fraud Detection",
    layout="wide",
    page_icon="🕵️‍♂️"
)


@st.cache_resource
def load_assets():

    data_path = PROCESSED_DIR / 'spektral_graph.pkl'
    weight_path_h5 = PROCESSED_DIR / 'gat_model.weights.h5'
    xgb_model_path = PROCESSED_DIR / 'xgb_model.json'
    ensemble_weights_path = PROCESSED_DIR / 'ensemble_weights.pkl'
    cleaned_data_path = PROCESSED_DIR / 'cleaned_transactions.parquet'

    if not data_path.exists():
        return None, None, None, None, None, None

    with open(data_path, 'rb') as f:
        graph = pickle.load(f)

    df_clean = pd.read_parquet(cleaned_data_path)

    model = build_model(num_features=graph.x.shape[1])

    coo = graph.a.tocoo()

    indices = np.column_stack([coo.row, coo.col])

    a_sparse = tf.SparseTensor(
        indices,
        coo.data.astype(np.float32),
        coo.shape
    )

    a_sparse = tf.sparse.reorder(a_sparse)

    x_tensor = tf.convert_to_tensor(
        graph.x,
        dtype=tf.float32
    )

    _ = model([x_tensor, a_sparse], training=False)

    model.load_weights(str(weight_path_h5))

    xgb_model = xgb.Booster()

    xgb_model.load_model(xgb_model_path)

    with open(ensemble_weights_path, 'rb') as f:
        ensemble_weights = pickle.load(f)

    return (
        graph,
        model,
        a_sparse,
        xgb_model,
        ensemble_weights,
        df_clean
    )


assets = load_assets()

if assets[0] is None:
    st.error("Graph data not found.")
    st.stop()

graph, model, a_sparse, xgb_model, ensemble_weights, df_clean = assets

st.title("🕵️‍♂️ Advanced Fraud Detection with GNN")

fraud_count = int(np.sum(graph.y))
total_count = len(graph.y)

st.markdown(
    f"""
    **Total Transactions**: {total_count:,}
    
    **Historical Fraud**: {fraud_count:,}
    ({fraud_count/total_count*100:.2f}%)
    """
)

st.divider()

tab1, tab2 = st.tabs([
    "📚 Historical Analysis",
    "⚡ Live Transaction"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("🔍 Historical Transaction Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:

        node_id = st.number_input(
            "Transaction Node ID",
            min_value=0,
            max_value=total_count - 1,
            value=0,
            step=1
        )

        if st.button("Analyze Transaction"):

            with st.spinner("Generating explanation..."):

                x_tensor = tf.convert_to_tensor(
                    graph.x,
                    dtype=tf.float32
                )

                explanation = explain_transaction(
                    model,
                    x_tensor,
                    a_sparse,
                    node_id
                )

                logits = model(
                    [x_tensor, a_sparse],
                    training=False
                )

                prob_gnn = tf.nn.sigmoid(
                    logits[node_id, 0]
                ).numpy()

                node_features = graph.x[node_id:node_id + 1]

                prob_xgb = float(
                    xgb_model.predict(
                        xgb.DMatrix(node_features)
                    )[0]
                )

                prob_blend = (
                    ensemble_weights['xgb_weight'] * prob_xgb
                    +
                    ensemble_weights['gnn_weight'] * prob_gnn
                )

                label = (
                    "Fraud"
                    if graph.y[node_id] == 1
                    else "Clean"
                )

                st.success("Analysis Complete")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Ensemble",
                    f"{prob_blend*100:.2f}%"
                )

                m2.metric(
                    "XGBoost",
                    f"{prob_xgb*100:.2f}%"
                )

                m3.metric(
                    "GNN",
                    f"{prob_gnn*100:.2f}%"
                )

                st.metric("True Label", label)

            with st.spinner("Rendering graph..."):

                html_path = str(
                    ROOT_DIR / 'app' / 'subgraph.html'
                )

                render_subgraph(
                    graph,
                    node_idx=node_id,
                    explanation=explanation,
                    hops=1,
                    output_path=html_path
                )

                st.session_state['html_path'] = html_path

    with col2:

        if (
            'html_path' in st.session_state
            and
            Path(st.session_state['html_path']).exists()
        ):

            with open(
                st.session_state['html_path'],
                'r',
                encoding='utf-8'
            ) as f:

                source_code = f.read()

            components.html(
                source_code,
                height=650
            )

        else:
            st.info("Run analysis to visualize graph.")

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("⚡ Live Fraud Prediction")

    col_a, col_b = st.columns([1, 2])

    with col_a:

        if 'live_tx' not in st.session_state:

            test_start = int(len(df_clean) * 0.85)

            st.session_state['live_tx'] = (
                df_clean.iloc[
                    random.randint(
                        test_start,
                        len(df_clean) - 1
                    )
                ].to_dict()
            )

        if st.button("🔄 Random Transaction"):

            test_start = int(len(df_clean) * 0.85)

            st.session_state['live_tx'] = (
                df_clean.iloc[
                    random.randint(
                        test_start,
                        len(df_clean) - 1
                    )
                ].to_dict()
            )

        live_tx = st.session_state['live_tx']

        st.markdown("### Modify Transaction")

        new_amt = st.number_input(
            "Transaction Amount",
            value=float(live_tx['TransactionAmt'])
        )

        new_card1 = st.number_input(
            "Card1",
            value=int(live_tx['card1'])
        )

        new_device = st.number_input(
            "DeviceInfo",
            value=int(live_tx['DeviceInfo'])
        )

        live_tx['TransactionAmt'] = new_amt
        live_tx['card1'] = new_card1
        live_tx['DeviceInfo'] = new_device

        if st.button("🚀 Process Transaction"):

            with st.spinner("Running inference..."):

                feature_cols = [
                    c for c in df_clean.columns
                    if c != 'isFraud'
                ]

                new_features = np.array(
                    [[live_tx[c] for c in feature_cols]],
                    dtype=np.float32
                )

                new_node_id = len(df_clean)

                edge_targets = set()

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

                for attr in EDGE_ATTRIBUTES:

                    val = live_tx[attr]

                    matches = df_clean.index[
                        df_clean[attr] == val
                    ].tolist()

                    matches = matches[:100]

                    for m in matches:
                        edge_targets.add(m)

                coo = graph.a.tocoo()

                rows = list(coo.row)
                cols = list(coo.col)
                data = list(coo.data)

                for target in edge_targets:

                    rows.append(new_node_id)
                    cols.append(target)
                    data.append(1.0)

                    rows.append(target)
                    cols.append(new_node_id)
                    data.append(1.0)

                rows.append(new_node_id)
                cols.append(new_node_id)
                data.append(1.0)

                new_shape = (
                    new_node_id + 1,
                    new_node_id + 1
                )

                indices = np.column_stack([rows, cols])

                new_a_sparse = tf.SparseTensor(
                    indices,
                    np.array(data, dtype=np.float32),
                    new_shape
                )

                new_a_sparse = tf.sparse.reorder(
                    new_a_sparse
                )

                new_x_tensor = tf.concat(
                    [
                        tf.convert_to_tensor(
                            graph.x,
                            dtype=tf.float32
                        ),
                        new_features
                    ],
                    axis=0
                )

                logits = model(
                    [new_x_tensor, new_a_sparse],
                    training=False
                )

                prob_gnn = tf.nn.sigmoid(
                    logits[new_node_id, 0]
                ).numpy()

                prob_xgb = float(
                    xgb_model.predict(
                        xgb.DMatrix(new_features)
                    )[0]
                )

                prob_blend = (
                    ensemble_weights['xgb_weight'] * prob_xgb
                    +
                    ensemble_weights['gnn_weight'] * prob_gnn
                )

                st.session_state['live_results'] = {
                    'connections': len(edge_targets),
                    'prob_blend': prob_blend,
                    'prob_xgb': prob_xgb,
                    'prob_gnn': prob_gnn
                }

                new_y = np.append(graph.y, [2])

                new_a_coo = sp.coo_matrix(
                    (data, (rows, cols)),
                    shape=new_shape
                )

                new_graph = Graph(
                    x=new_x_tensor.numpy(),
                    a=new_a_coo.tocsr(),
                    y=new_y
                )

                html_path = str(
                    ROOT_DIR / 'app' / 'live_subgraph.html'
                )

                render_subgraph(
                    new_graph,
                    node_idx=new_node_id,
                    explanation=None,
                    hops=1,
                    output_path=html_path
                )

                st.session_state['live_html_path'] = html_path

    with col_b:

        if 'live_results' in st.session_state:

            res = st.session_state['live_results']

            st.success(
                f"Connected with {res['connections']} historical transactions"
            )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Ensemble",
                f"{res['prob_blend']*100:.2f}%"
            )

            m2.metric(
                "XGBoost",
                f"{res['prob_xgb']*100:.2f}%"
            )

            m3.metric(
                "GNN",
                f"{res['prob_gnn']*100:.2f}%"
            )

            if (
                'live_html_path' in st.session_state
                and
                Path(st.session_state['live_html_path']).exists()
            ):

                with open(
                    st.session_state['live_html_path'],
                    'r',
                    encoding='utf-8'
                ) as f:

                    source_code = f.read()

                components.html(
                    source_code,
                    height=500
                )

        else:
            st.info("Process a transaction to visualize network.")