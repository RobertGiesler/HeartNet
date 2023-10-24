import pandas as pd


def knn_graph_from_adjacency_matrix(
    adj_matrix_df, k, directed=False, weights="distance", include_self=False
):
    """
    Calculates the k-nearest neighbor graph from a nxn adjacency matrix.

    Parameters
    ----------
    adj_matrix_df : pandas DataFrame
        Adjacency matrix of the graph. Entry [i,j] is the weight of the edge going from
        j to i (equal to the weight of the edge from i to j for undirected graphs).
    k : int
        Number of nearest neighbors to connect each node to. For undirected graphs,
        nearest neighbors may not be symmetrical. In this case, each node is connected
        to its k nearest neighbors and some nodes may have more than k neighbors in the
        resulting graph.
    directed : bool
        Whether the input graph is directed. Default is `False`.
    weights : 'distance' or 'similarity'
        Whether edge weights represent distance or similarity between nodes. Default is
        `'distance'`.
    include_self : bool
        Whether or not to consider self loops. If `False`, self loops will be maintained
        but are not counted as a neighbor. If `True`, self loops are also considered as
        neighbors. Default is `False`.

    Returns
    -------
    pandas DataFrame (dtype = float)
        Adjacency matrix of the k-nearest neighbor graph of the input graph. Weights of
        edges of non-connected nodes are set to 0.
    """
    # Sort adjacency matrix by columns and index
    adj_matrix_df.sort_index(axis="columns", inplace=True)
    adj_matrix_df.sort_index(axis="index", inplace=True)
    columns = adj_matrix_df.columns
    index = adj_matrix_df.index
    # Check that rows and columns are equal
    if not columns.equals(index):
        print(f"index: {index}")
        print(f"columns: {columns}")
        raise ValueError(
            "[ERROR] Invalid adjacency matrix. Rows and columns of adj_matrix_df must"
            " be equal!"
        )

    # Each node has k or less edges
    if k >= len(index):
        return adj_matrix_df

    adjacency_dict = {}
    for i in columns:
        adjacency_dict[i] = {}

    for current_label in adj_matrix_df:
        column = adj_matrix_df[current_label]
        sorting_indices = column.argsort()

        # If weights represent similarity, flip the sorting indices
        if weights not in ["distance", "similarity"]:
            raise ValueError(
                f"[ERROR] weights must be 'distance' or 'similarity' but is"
                f" '{weights}'."
            )
        if weights == "similarity":
            sorting_indices = sorting_indices.reindex(index=sorting_indices.index[::-1])

        additional_neighbor = False  # To keep track of self loop
        # Set edge values in adjacency_dict for k nearest neighbors of each node
        for i in range(k):
            neighbor_index = sorting_indices.iloc[i]
            neighbor_label = column.index[neighbor_index]
            edge_value = column.iloc[neighbor_index]
            # If include_self == False, we don't want to add own column label
            if not include_self and neighbor_label == current_label:
                additional_neighbor = True
                continue
            adjacency_dict[current_label][neighbor_label] = edge_value

        # We need to add one more neighbor if we counted current_label as a neighbor
        if additional_neighbor:
            neighbor_index = sorting_indices.iloc[k]
            neighbor_label = column.index[neighbor_index]
            edge_value = column.iloc[neighbor_index]
            adjacency_dict[current_label][neighbor_label] = edge_value

    # If graph is not directed, make sure adjacency_dict is symmetrical
    if not directed:
        sym_adj_dict = adjacency_dict
        for root, targets in adjacency_dict.items():
            for target, weight in targets.items():
                if not root in sym_adj_dict[target]:
                    sym_adj_dict[target][root] = weight
        adjacency_dict = sym_adj_dict

    # Fill up adjacency dict with zeros
    for i in index:
        for root, targets in adjacency_dict.items():
            if i not in targets:
                adjacency_dict[root][i] = 0.0

    # Create new dataframe
    knn_df = pd.DataFrame(adjacency_dict, dtype=float)

    # Sort rows of dataframe
    knn_df.sort_index(axis="index", inplace=True)

    return knn_df
