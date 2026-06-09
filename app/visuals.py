import networkx as nx
from pyvis.network import Network
import numpy as np
import scipy.sparse as sp


def get_k_hop_subgraph(a_matrix, node_idx, hops=1, max_nodes=200):

    if not sp.isspmatrix_csr(a_matrix):
        a_matrix = a_matrix.tocsr()

    visited = set([node_idx])
    frontier = set([node_idx])

    for _ in range(hops):

        next_frontier = set()

        for node in frontier:

            neighbors = a_matrix[node].indices

            next_frontier.update(neighbors)

        next_frontier -= visited

        visited.update(next_frontier)

        frontier = next_frontier

        if len(visited) >= max_nodes:
            break

    subset_nodes = np.array(list(visited))

    if subset_nodes.size == 0:
        return sp.csr_matrix((0, 0)), np.array([], dtype=int)

    subset_nodes = subset_nodes[:max_nodes]

    sub_a = a_matrix[subset_nodes][:, subset_nodes]

    return sub_a, subset_nodes


def render_subgraph(graph, node_idx, explanation=None, hops=1, output_path="subgraph.html"):

    sub_a, subset_nodes = get_k_hop_subgraph(
        graph.a,
        node_idx,
        hops=hops
    )

    if subset_nodes.size == 0:

        net = Network(
            height="600px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False
        )

        color = '#ff4d4d' if graph.y[node_idx] == 1 else '#4CAF50'

        net.add_node(
            0,
            label=f"Txn {node_idx}",
            color=color,
            size=30,
            shape='star'
        )

        net.save_graph(output_path)

        return output_path

    G = nx.from_scipy_sparse_array(sub_a)

    labels = graph.y[subset_nodes]

    for i, orig_idx in enumerate(subset_nodes):

        if labels[i] == 1:
            color = '#ff4d4d'

        elif labels[i] == 0:
            color = '#4CAF50'

        else:
            color = '#3b82f6'

        size = 30 if orig_idx == node_idx else 15

        shape = 'star' if orig_idx == node_idx else 'dot'

        G.nodes[i]['label'] = f"Txn {orig_idx}"
        G.nodes[i]['color'] = color
        G.nodes[i]['size'] = size
        G.nodes[i]['shape'] = shape

    for u, v in G.edges():

        G[u][v]['color'] = '#cccccc'
        G[u][v]['width'] = 1

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False
    )

    net.from_nx(G)

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    net.save_graph(output_path)

    return output_path