from heartnet import data_preprocessing
import pandas as pd
from pathlib import Path
import math
import networkx as nx
import henhoe2vec as hh2v


def test_adj_to_edg_list_undirected(tmp_path):
    # ----------------------------------------------------------------------------------
    # PREPARE TEST ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_dir = Path(tmp_path).joinpath("input/")
    Path.mkdir(input_dir)

    # layer1 adjacency matrix
    l1 = [
        [0, 2, 4, 0],
        [2, 0, 5, 9],
        [4, 5, 0, 0],
        [0, 9, 0, 0],
    ]
    l1_index = ["n1", "n2", "n3", "n4"]
    l1_df = pd.DataFrame(l1, index=l1_index, columns=l1_index)
    save_path = input_dir.joinpath("_layer_l1.csv")
    # save layer1 adjacency matrix
    l1_df.to_csv(save_path, sep=",")

    # layer2 adjacency matrix
    l2 = [
        [0.0, 1.3, 0.0, 2.8, 0.0],
        [1.3, 0.0, 5.5, 9.7, 2.0],
        [0.0, 5.5, 0.0, 0.0, 0.0],
        [2.8, 9.7, 0.0, 0.0, 4.2],
        [0.0, 2.0, 0.0, 4.2, 0.0],
    ]
    l2_index = ["1", "2", "3", "4", "5"]
    l2_df = pd.DataFrame(l2, index=l2_index, columns=l2_index)
    save_path = input_dir.joinpath("_layer_l2.csv")
    # save layer2 adjacency matrix
    l2_df.to_csv(save_path, sep=",")

    # inter-layer adjacency matrix
    inter = [
        [0, 0, 1, 0, 0],
        [1, 2, 0, 0, 3],
        [0, 1, 0, 0, 2],
        [3, 0, 0, 0, 1],
    ]
    inter_index = ["n1", "n2", "n3", "n4"]
    inter_columns = ["1", "2", "3", "4", "5"]
    inter_df = pd.DataFrame(inter, index=inter_index, columns=inter_columns)
    save_path = input_dir.joinpath("_interlayer_l1_l2.csv")
    # save interlayer adjacency matrix
    inter_df.to_csv(save_path, sep=",")

    num_nodes = 9
    num_edges = 4 + 6 + 8  # Undirected edges

    # ----------------------------------------------------------------------------------
    # CONVERT ADJACENCY MATRICES INTO MULTILAYER EDGE LIST
    # ----------------------------------------------------------------------------------
    output_dir = Path(tmp_path).joinpath("output/")
    Path.mkdir(output_dir)
    output_file = output_dir.joinpath("multi_edgelist.edg")

    data_preprocessing.adjacency_matrices_to_multilayer_edgelist(
        input_dir=input_dir,
        output_file=output_file,
        directed=False,
        input_sep=",",
        output_header=True,
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    result_df = pd.read_csv(output_file, sep=",")
    # Check header
    target_header = pd.Index(
        ["source", "source_layer", "target", "target_layer", "weight"]
    )
    assert result_df.columns.equals(target_header)
    # Check that there are no edges with weight == 0.0
    assert len(result_df[result_df.weight == 0.0]) == 0

    # Parse multilayer edge list as NetworkX graph and check nodes and edges
    N = hh2v.utils.parse_multilayer_edgelist(
        output_file, directed=False, sep=",", header=True
    )
    assert N.number_of_nodes() == num_nodes
    assert N.number_of_edges() == num_edges
    assert ("n1", "l1") in N.nodes
    assert (("n4", "l1"), ("2", "l2")) not in N.edges
    assert (("5", "l2"), ("n2", "l1")) in N.edges
    assert (("n2", "l1"), ("5", "l2")) in N.edges
    assert N[("5", "l2")][("n2", "l1")]["weight"] == 3.0
    assert N[("n1", "l1")][("n3", "l1")]["weight"] == 4.0
    assert N.degree(("n1", "l1")) == 3


def test_adj_to_edg_list_directed(tmp_path):
    # ----------------------------------------------------------------------------------
    # PREPARE TEST ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_dir = Path(tmp_path).joinpath("input/")
    Path.mkdir(input_dir)

    # layer1 adjacency matrix
    l1 = [
        [0, 0, 0, 0.7],
        [2, 0, 5.2, 9],
        [8, 1.3, 0, 0],
        [0, 9, 3, 0],
    ]
    l1_index = ["n1", "n2", "n3", "n4"]
    l1_df = pd.DataFrame(l1, index=l1_index, columns=l1_index)
    save_path = input_dir.joinpath("_layer_l1.csv")
    # save layer1 adjacency matrix
    l1_df.to_csv(save_path, sep="\t")

    # layer2 adjacency matrix
    l2 = [
        [0.0, 1.3, 0.0, 2.8, 0.0],
        [0.0, 0.0, 5.5, 9.0, 3.5],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [2.8, 9.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.2, 0.0],
    ]
    l2_index = ["1", "2", "3", "4", "5"]
    l2_df = pd.DataFrame(l2, index=l2_index, columns=l2_index)
    save_path = input_dir.joinpath("_layer_l2.csv")
    # save layer2 adjacency matrix
    l2_df.to_csv(save_path, sep="\t")

    # l1-l2 inter-layer adjacency matrix
    inter = [
        [0, 0, 1, 0, 0],
        [1, 2, 0, 0, 0],
        [0, 1, 0, 0, 2],
        [3, 0, 0, 0, 1],
    ]
    inter_index = ["n1", "n2", "n3", "n4"]
    inter_columns = ["1", "2", "3", "4", "5"]
    inter_df = pd.DataFrame(inter, index=inter_index, columns=inter_columns)
    save_path = input_dir.joinpath("_interlayer_l1_l2.csv")
    # save l1-l2 interlayer adjacency matrix
    inter_df.to_csv(save_path, sep="\t")

    # l2-l1 inter-layer adjacency matrix
    inter = [
        [5, 0, 1, 0],
        [0, 0, 0, 3],
        [0, 3, 0, 4],
        [9, 0, 3, 0],
        [0, 3, 5, 3],
    ]
    inter_columns = ["n1", "n2", "n3", "n4"]
    inter_index = ["1", "2", "3", "4", "5"]
    inter_df = pd.DataFrame(inter, index=inter_index, columns=inter_columns)
    save_path = input_dir.joinpath("_interlayer_l2_l1.csv")
    # save l2-l1 interlayer adjacency matrix
    inter_df.to_csv(save_path, sep="\t")

    num_nodes = 9
    num_edges = 8 + 9 + 7 + 10  # Directed edges

    # ----------------------------------------------------------------------------------
    # CONVERT ADJACENCY MATRICES INTO MULTILAYER EDGE LIST
    # ----------------------------------------------------------------------------------
    output_dir = Path(tmp_path).joinpath("output/")
    Path.mkdir(output_dir)
    output_file = output_dir.joinpath("multi_edgelist.edg")

    data_preprocessing.adjacency_matrices_to_multilayer_edgelist(
        input_dir=input_dir,
        output_file=output_file,
        directed=True,
        input_sep="\t",
        output_sep="\t",
        output_header=False,
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    result_df = pd.read_csv(output_file, sep="\t")
    # Check that there are 5 columns
    assert len(result_df.columns) == 5
    # Check that there are no edges with weight == 0.0
    assert len(result_df[result_df.iloc[:, 4] == 0.0]) == 0

    # Parse multilayer edge list as NetworkX graph and check nodes and edges
    N = hh2v.utils.parse_multilayer_edgelist(
        output_file, directed=True, sep="\t", header=False
    )
    assert N.number_of_nodes() == num_nodes
    assert N.number_of_edges() == num_edges
    assert ("n1", "l1") in N.nodes

    assert (("1", "l2"), ("2", "l2")) in N.edges
    assert N[("1", "l2")][("2", "l2")]["weight"] == 1.3

    assert (("n2", "l1"), ("5", "l2")) not in N.edges
    assert (("5", "l2"), ("n2", "l1")) in N.edges
    assert N[("5", "l2")][("n2", "l1")]["weight"] == 3.0

    assert (("n3", "l1"), ("n1", "l1")) in N.edges
    assert N[("n3", "l1")][("n1", "l1")]["weight"] == 8.0
    assert (("n1", "l1"), ("n3", "l1")) not in N.edges

    assert N.out_degree(("n1", "l1")) == 2


def test_interlayer_k_neighbors_axis_1_distance():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_matrix = [
        [2.4, 5, 2, 1],
        [5.5, 9, 3, 4],
        [0, 9, 4, 0.2],
        [3, 4, 5, 6.0],
        [7, 3.4, 5, 2],
    ]
    columns = ["A", "B", "C", "D"]
    index = [1, 2, 3, 4, 5]
    input_df = pd.DataFrame(input_matrix, columns=columns, index=index)
    neighbors = 3  # Connect each node to its 3 nearest neighbors

    target_matrix = [
        [2.4, 5.0, 2.0, 1.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0, 0.2],
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 3.4, 0.0, 2.0],
    ]
    columns = ["A", "B", "C", "D"]
    index = [1, 2, 3, 4, 5]
    target_df = pd.DataFrame(target_matrix, columns=columns, index=index)

    # ----------------------------------------------------------------------------------
    # RUN K-NEIGHBORS
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.interlayer_k_neighbors(
        input_df, neighbors, weights="distance", axis=1
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    print(f"result: {result_df}")
    print(f"target: {target_df}")
    assert result_df.equals(target_df)


def test_interlayer_k_neighbors_axis_0_similarity():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_matrix = [
        [2, 5, 6, 2, 7],
        [0, 0, 0, 0, 0],
        [2, 4, 1, 0, 9],
        [6, 3, 5, 4, 3],
    ]
    columns = ["A", "B", "C", "D", "E"]
    index = [1, 2, 3, 4]
    input_df = pd.DataFrame(input_matrix, columns=columns, index=index)
    neighbors = 2  # Connect each node to its 2 nearest neighbors

    target_matrix = [[0, 0, 6, 0, 7], [0, 0, 0, 0, 0], [0, 4, 0, 0, 9], [6, 0, 5, 0, 0]]
    columns = ["A", "B", "C", "D", "E"]
    index = [1, 2, 3, 4]
    target_df = pd.DataFrame(target_matrix, columns=columns, index=index, dtype=float)

    # ----------------------------------------------------------------------------------
    # RUN K-NEIGHBORS
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.interlayer_k_neighbors(
        input_df, neighbors, weights="similarity", axis=0
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    print(f"result: {result_df}")
    print(f"target: {target_df}")
    assert result_df.equals(target_df)


def test_invert_weights_adjacency_matrix(tmp_path):
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    input_dir = Path(tmp_path).joinpath("input/")
    Path.mkdir(input_dir)

    # inter-layer adjacency matrix
    inter = [
        [0, 0, 1, 0, 0],
        [1, 2, 0, 0, 3],
        [0, 1, 0, 0, 2],
        [3, 0, 0, 0, 1],
    ]
    inter_index = ["n1", "n2", "n3", "n4"]
    inter_columns = ["1", "2", "3", "4", "5"]
    inter_df = pd.DataFrame(inter, index=inter_index, columns=inter_columns)
    infput_file = input_dir.joinpath("_interlayer_l1_l2.csv")
    # save interlayer adjacency matrix
    inter_df.to_csv(infput_file, sep=",")

    # target adjacency matrix
    target = [
        [0, 0, 1, 0, 0],
        [1, 1 / 2, 0, 0, 1 / 3],
        [0, 1, 0, 0, 1 / 2],
        [1 / 3, 0, 0, 0, 1],
    ]
    target_index = ["n1", "n2", "n3", "n4"]
    target_columns = ["1", "2", "3", "4", "5"]
    target_df = pd.DataFrame(
        target, index=target_index, columns=target_columns, dtype=float
    )

    # ----------------------------------------------------------------------------------
    # INVERT WEIGHTS OF ADJACENCY MATRIX
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.invert_weights_adjacency_matrix(infput_file, sep=",")

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    assert result_df.equals(target_df)


def test_min_max_normalization_by_column():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET DATAFRAME
    # ----------------------------------------------------------------------------------
    test_df = [[1, 4, 5], [0, 2, 1], [2, 1, 0]]
    columns = ["A", "B", "C"]
    index = [2, 3, 4]
    test_df = pd.DataFrame(test_df, index=index, columns=columns)

    target_df = [[0.5, 1, 1], [0, 1 / 3, 0.2], [1, 0, 0]]
    columns = ["A", "B", "C"]
    index = [2, 3, 4]
    target_df = pd.DataFrame(target_df, index=index, columns=columns, dtype=float)

    # ----------------------------------------------------------------------------------
    # NORMALIZE
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.min_max_normalization_by_column(test_df)

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    assert result_df.equals(target_df)


def test_min_max_normalization_by_column_custom_range():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET DATAFRAME
    # ----------------------------------------------------------------------------------
    test_df = [[2, 0, 4], [3, 0, 8], [0, 0, 2]]
    columns = ["A", "B", "C"]
    index = [0, 1, 2]
    test_df = pd.DataFrame(test_df, index=index, columns=columns)

    range = [1, 11]

    target_df = [[20 / 3 + 1, 1, 20 / 6 + 1], [11, 1, 11], [1, 1, 1]]
    columns = ["A", "B", "C"]
    index = [0, 1, 2]
    target_df = pd.DataFrame(target_df, index=index, columns=columns, dtype=float)

    # ----------------------------------------------------------------------------------
    # NORMALIZE
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.min_max_normalization_by_column(test_df, range=range)

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    assert result_df.equals(target_df)


def test_min_max_normalization():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET DATAFRAME
    # ----------------------------------------------------------------------------------
    test_df = [[2, 4, 3], [1, 5, 7], [0, 0, 2]]
    columns = ["A", "B", "C"]
    index = [2, 3, 4]
    test_df = pd.DataFrame(test_df, index=index, columns=columns)

    target_df = [[2 / 7, 4 / 7, 3 / 7], [1 / 7, 5 / 7, 1], [0, 0, 2 / 7]]
    columns = ["A", "B", "C"]
    index = [2, 3, 4]
    target_df = pd.DataFrame(target_df, index=index, columns=columns, dtype=float)

    # ----------------------------------------------------------------------------------
    # NORMALIZE
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.min_max_normalization(test_df)

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    # Round all values to 6 decimal places to avoid failure due to rounding errors
    assert result_df.round(6).equals(target_df.round(6))


def test_min_max_normalization_custom_range():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET DATAFRAME
    # ----------------------------------------------------------------------------------
    test_df = [[3, 2, 1, 9], [4, 1, 2, 2]]
    columns = ["A", "B", "C", "D"]
    index = [0, 1]
    test_df = pd.DataFrame(test_df, index=index, columns=columns)

    range = [2, 5]

    target_df = [[6 / 8 + 2, 3 / 8 + 2, 2, 5], [9 / 8 + 2, 2, 3 / 8 + 2, 3 / 8 + 2]]
    columns = ["A", "B", "C", "D"]
    index = [0, 1]
    target_df = pd.DataFrame(target_df, index=index, columns=columns, dtype=float)

    # ----------------------------------------------------------------------------------
    # NORMALIZE
    # ----------------------------------------------------------------------------------
    result_df = data_preprocessing.min_max_normalization(test_df, range=range)

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target: {target_df}")
    print(f"result: {result_df}")
    # Round all values to 6 decimal places to avoid failure due to rounding errors
    assert result_df.round(6).equals(target_df.round(6))


def test_interlayer_mean_and_std():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST ADJACENCY MATRIX AND TARGET MEAN AND STD
    # ----------------------------------------------------------------------------------
    layer1 = {
        "n1": [0, 3, -2, 0.5],
        "n2": [4, 2, 0, 5],
        "n3": [0, 0, 1, -1],
        "n4": [2, 4, 1, 0],
    }
    layer2 = ["u1", "u2", "u3", "u4"]

    adj_matrix = pd.DataFrame.from_dict(layer1, orient="index", columns=layer2)

    target_mean = 1.21875  # Calculated by hand

    values = adj_matrix.to_numpy().flatten()
    n = len(values)
    diff_squared = [math.pow(i - target_mean, 2) for i in values]
    target_std = math.sqrt(sum(diff_squared) / (n - 1))

    # ----------------------------------------------------------------------------------
    # CALCULATE MEAN AND STD
    # ----------------------------------------------------------------------------------
    mean, std = data_preprocessing.get_interlayer_weight_mean_and_std(adj_matrix)

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target mean: {target_mean}, target std: {target_std}")
    print(f"result mean: {mean}, result std: {std}")
    assert mean == target_mean and std == target_std


def test_interlayer_threshold_below_inplace():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    layer1 = {
        "n1": [0, 3, -2, 0.5, 100],
        "n2": [4, 2, 0, 5, -100],
        "n3": [0, 0, 1, -1, 0],
        "n4": [2, 4, 1, 0, 2000],
    }
    layer2 = ["u1", "u2", "u3", "u4", "u5"]
    adj_matrix = pd.DataFrame.from_dict(
        layer1, orient="index", columns=layer2, dtype=float
    )

    threshold = 3

    target_layer1 = {
        "n1": [0, 3, 0, 0, 100],
        "n2": [4, 0, 0, 5, 0],
        "n3": [0, 0, 0, 0, 0],
        "n4": [0, 4, 0, 0, 2000],
    }
    target_layer2 = ["u1", "u2", "u3", "u4", "u5"]
    target_matrix = pd.DataFrame.from_dict(
        target_layer1, orient="index", columns=target_layer2, dtype=float
    )

    # ----------------------------------------------------------------------------------
    # THRESHOLD ADJACENCY MATRIX
    # ----------------------------------------------------------------------------------
    new_adj_matrix = data_preprocessing.interlayer_threshold(
        adj_matrix, threshold=threshold, remove="below", inplace=True
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    # new_adj_matrix should be None because inplace=True
    assert not new_adj_matrix

    print(f"target: \n{target_matrix}")
    print(f"result: \n{adj_matrix}")
    assert adj_matrix.equals(target_matrix)


def test_interlayer_threshold_above_not_inplace():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST AND TARGET ADJACENCY MATRICES
    # ----------------------------------------------------------------------------------
    layer1 = {
        "n1": [0, 3, -2, 0.5, 100],
        "n2": [4, 2, 0, 5, -100],
        "n3": [0, 0, 1, -1, 0],
        "n4": [2, 4, 1, 0, 2000],
    }
    layer2 = ["u1", "u2", "u3", "u4", "u5"]
    adj_matrix = pd.DataFrame.from_dict(
        layer1, orient="index", columns=layer2, dtype=float
    )

    threshold = 3

    target_layer1 = {
        "n1": [0, 3, -2, 0.5, 0],
        "n2": [0, 2, 0, 0, -100],
        "n3": [0, 0, 1, -1, 0],
        "n4": [2, 0, 1, 0, 0],
    }
    target_layer2 = ["u1", "u2", "u3", "u4", "u5"]
    target_matrix = pd.DataFrame.from_dict(
        target_layer1, orient="index", columns=target_layer2, dtype=float
    )

    # ----------------------------------------------------------------------------------
    # THRESHOLD ADJACENCY MATRIX
    # ----------------------------------------------------------------------------------
    new_adj_matrix = data_preprocessing.interlayer_threshold(
        adj_matrix, threshold=threshold, remove="above", inplace=False
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULT
    # ----------------------------------------------------------------------------------
    # adj_matrix should have original value because inplace=False
    original_matrix = pd.DataFrame.from_dict(
        layer1, orient="index", columns=layer2, dtype=float
    )
    assert adj_matrix.equals(original_matrix)

    print(f"target: \n{target_matrix}")
    print(f"result: \n{new_adj_matrix}")
    assert new_adj_matrix.equals(target_matrix)
