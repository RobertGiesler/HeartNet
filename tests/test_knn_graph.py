from heartnet import knn_graph
import pandas as pd
import pytest


def test_knn_graph_from_adj_matrix_distance():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_matrix = [[2, 3, 5, 6], [3, 5, 2, 7], [5, 2, 8, 1], [6, 7, 1, 9]]
    index = [3, 4, 5, 6]
    input_df = pd.DataFrame(input_matrix, index=index, columns=index)

    # include_self = False
    target_matrix = [
        [0, 3, 5, 6],
        [3, 0, 2, 0],
        [5, 2, 0, 1],
        [6, 0, 1, 0],
    ]
    index = [3, 4, 5, 6]
    target_df = pd.DataFrame(target_matrix, index=index, columns=index, dtype=float)

    k = 2

    # ----------------------------------------------------------------------------------
    # COMPUTE KNN GRAPH
    # ----------------------------------------------------------------------------------
    result_df = knn_graph.knn_graph_from_adjacency_matrix(
        input_df, k=k, directed=False, weights="distance", include_self=False
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    assert result_df.equals(target_df)


def test_knn_graph_from_adj_matrix_similarity_include_self():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_matrix = [
        [2, 3, 5, 6],
        [3, 5, 2, 7],
        [5, 2, 8, 1],
        [6, 7, 1, 9],
    ]
    index = ["A", "B", "C", "D"]
    input_df = pd.DataFrame(input_matrix, index=index, columns=index)

    # include_self = False
    target_matrix = [
        [0, 0, 5, 6],
        [0, 5, 0, 7],
        [5, 0, 8, 0],
        [6, 7, 0, 9],
    ]
    index = ["A", "B", "C", "D"]
    target_df = pd.DataFrame(target_matrix, index=index, columns=index, dtype=float)

    k = 2

    # ----------------------------------------------------------------------------------
    # COMPUTE KNN GRAPH
    # ----------------------------------------------------------------------------------
    result_df = knn_graph.knn_graph_from_adjacency_matrix(
        input_df, k=k, directed=False, weights="similarity", include_self=True
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    assert result_df.equals(target_df)


def test_knn_graph_from_adj_matrix_invalid_weights():
    input_matrix = [
        [2, 3, 5, 6],
        [3, 5, 2, 7],
        [5, 2, 8, 1],
        [6, 7, 1, 9],
    ]
    index = ["A", "B", "C", "D"]
    input_df = pd.DataFrame(input_matrix, index=index, columns=index)

    with pytest.raises(ValueError):
        result = knn_graph.knn_graph_from_adjacency_matrix(
            input_df, k=2, weights="invalid"
        )
