import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from . import knn_graph
import shutil


def remove_cellsubtype2(in_file, save_file=False, output_file=None, overwrite=False):
    """
    Read the .csv file of an adjacency matrix and drop the cell_subtype2 column.

    Parameters
    ----------
    in_file : str or pathlib.Path object
        Path of interlayer spot-celltype csv file.
    save_file : bool
        Whether to save the resulting adjacency matrix as a .csv file. Default is False.
    output_file : str or pathlib.Path object
        Path of the output .csv file. If `out_file` is a directory, the output file will
        be saved with the same name as the input file. Default is None.
    overwrite : bool
        Whether to overwrite existing .csv file. Default is False.

    Returns
    -------
    pandas DataFrame
        The adjacency matrix without the cell_subtype2 column as a DataFrame.
    """
    in_path = Path(in_file)
    if not in_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file {in_path} does not exist.")

    adj_mtrx = pd.read_csv(in_path, index_col=0)
    adj_mtrx.drop(axis="columns", labels="cell_subtype2", inplace=True)

    if save_file:
        if not output_file:
            raise ValueError(
                f"[ERROR] output_file cannot be None if save_file is True."
            )
        out_path = Path(output_file)
        if out_path.is_dir():
            out_path = out_path.joinpath(in_path.name)
        if out_path.is_file() and not overwrite:
            raise FileExistsError(f"[ERROR] File {out_path} already exists.")

        if not out_path.exists():
            print(f"[STATUS] Created file {out_path}.", flush=True)
        if out_path.suffix == ".csv":
            adj_mtrx.to_csv(out_path)
        else:
            raise ValueError(f"[ERROR] output_file: {output_file} is not a .csv file.")

    return adj_mtrx


def adjacency_matrices_to_multilayer_edgelist(
    input_dir,
    output_file,
    directed=False,
    input_sep=",",
    output_sep=",",
    output_header=True,
    overwrite=False,
):
    """
    Convert intra- and inter-layer adjacency matrices into a multilayer edge list.

    Parameters
    ----------
    input_dir : str or pathlib.Path object
        Path of the directory containing the input adjacency matrices as CSV files
        (with header and index). The directory must contain an adjacency matrix for each
        layer and an inter-layer connection matrix for each pair of layers which are
        connected by inter-layer edges. If the network is directed, two inter-layer
        connection matrices are required for each layer pair - one for each direction.
        Layer adjacency matrices must be named according to the following pattern:
        `*_layer_X.csv` where `X` is the name of the layer. Inter-layer adjacency
        matrices must be named according to the following pattern:
        `*_interlayer_X_Y.csv` where `X` is the name of the layer that the nodes on the
        rows of the CSV belong to and `Y` is the name of the layer which the nodes on
        the columns of the CSV belong to. Avoid `_` in the layer names.
    output_file : str or pathlib.Path object
        Path of the output edge list CSV file. Consists of the columns 'source',
        'source_layer', 'target', 'target_layer', 'weight'.
    directed : bool
        Whether the network is directed or not. If directed, an entry (i,j) of the CSV
        files corresponds to an edge from i to j.
    input_sep : str
        Delimiter used in input CSV files. Default is ','.
    output_sep : str
        Delimiter to be used in the output CSV edge list. Default is ','.
    output_header : bool
        Whether the output CSV edge list has a header row. Default is False.
    overwrite : bool
        Whether to overwrite `output_file` if it already exists. Default is False.

    Returns
    -------
    pandas DataFrame
        The multiayer edge list as a DataFrame. Consists of the columns 'source',
        'source_layer', 'target', 'target_layer', 'weight'.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"[ERROR] input_dir: {input_dir} is not a directory.")

    output_file = Path(output_file)
    if output_file.exists():
        if overwrite:
            print(f"[STATUS] Overwriting {output_file}.", flush=True)
        else:
            raise FileExistsError(f"[ERROR] output_file: {output_file} already exists.")

    # Generator of paths of (intra-)layer adjacency matrices
    layer_paths = input_dir.glob("*_layer_*.csv")
    # Generator of path of inter-layer adjacency matrices
    inter_layer_paths = input_dir.glob("*_interlayer_*.csv")

    # List of pandas DataFrames which represent the intra- and inter-layer edge lists
    edgelist_dataframes = []

    for layer in layer_paths:
        # Extract the name of the layer
        layer_name = layer.stem.split("_layer_")[-1]
        print(f"[STATUS] Found adjacency matrix for layer {layer_name}.", flush=True)
        # Adjacency matrix as pandas DataFrame
        adj_df = pd.read_csv(layer, sep=input_sep, index_col=0)
        # Cast index as str (in case node names are integers)
        adj_df.index = adj_df.index.astype(str)
        # Turn adjacency matrix into NetworkX graph
        layer_nx = nx.from_pandas_adjacency(
            adj_df, create_using=nx.DiGraph if directed else nx.Graph
        )
        # Turn NetworkX graph into an edge list DataFrame
        layer_edgelist_df = nx.to_pandas_edgelist(layer_nx)
        # Append layer information
        layer_edgelist_df["source_layer"] = layer_name
        layer_edgelist_df["target_layer"] = layer_name

        # Add to list of intra-layer edge lists
        edgelist_dataframes.append(layer_edgelist_df)

    for inter_layer in inter_layer_paths:
        # Extract the names of the layers
        layer_names = inter_layer.stem.split("_interlayer_")[-1]
        source_layer_name = layer_names.split("_")[0]
        target_layer_name = layer_names.split("_")[1]
        print(
            f"[STATUS] Found inter-layer adjacency matrix between layers"
            f" {source_layer_name} and {target_layer_name}.",
            flush=True,
        )

        # Dict to store the inter-layer edge list information
        edgelist_dict = {
            "source": [],
            "target": [],
            "weight": [],
        }

        with open(inter_layer) as IN:
            for i, line in enumerate(IN):
                # First line (header)
                if i == 0:
                    # Get column names
                    header = line.strip("\n").split(sep=input_sep)

                # All other lines (matrix entries)
                else:
                    line = line.strip("\n").split(sep=input_sep)
                    for j, entry in enumerate(line):
                        # Index column
                        if j == 0:
                            # Source node
                            source = entry
                        # Edges
                        else:
                            target = header[j]
                            weight = entry
                            if float(weight) != 0.0:
                                edgelist_dict["source"].append(source)
                                edgelist_dict["target"].append(target)
                                edgelist_dict["weight"].append(weight)

        inter_edgelist_df = pd.DataFrame.from_dict(edgelist_dict)
        # Append layer information
        inter_edgelist_df["source_layer"] = source_layer_name
        inter_edgelist_df["target_layer"] = target_layer_name

        # Add to list of inter-layer edge lists
        edgelist_dataframes.append(inter_edgelist_df)

    # Concatenate all edge lists
    multiedgelist_df = pd.concat(edgelist_dataframes, ignore_index=True)
    # Save multilayer edge list
    multiedgelist_df.loc[
        :, ["source", "source_layer", "target", "target_layer", "weight"]
    ].to_csv(output_file, sep=output_sep, index=False, header=output_header)


def adjacency_matrices_to_celltype_edgelist(
    input_dir,
    output_file,
    directed=False,
    input_sep=",",
    output_sep=",",
    output_header=True,
    overwrite=False,
):
    """
    Convert intra- and inter-layer adjacency matrices into a cell type edge list. Spot
    and inter-layer adjacency matrices are ignored.

    Parameters
    ----------
    input_dir : str or pathlib.Path object
        Path of the directory containing the input adjacency matrices as .csv files
        (with header and index). The directory must contain at least one adjacency
        matrix csv file named according to the pattern `*_layer_celltype.csv`. If the
        network is directed, two inter-layer adjacency matrices are required - one for
        each direction.
    output_file : str or pathlib.Path object
        Path of the output edge list .csv file. Consists of the columns 'source',
        'source_layer', 'target', 'target_layer', 'weight'.
    directed : bool
        Whether the network is directed or not. If directed, an entry (i,j) of the .csv
        files corresponds to an edge from i to j.
    input_sep : str
        Delimiter used in input .csv files. Default is ','.
    output_sep : str
        Delimiter to be used in the output .csv edge list. Default is ','.
    output_header : bool
        Whether the output .csv edge list has a header row. Default is False.
    overwrite : bool
        Whether to overwrite `output_file` if it already exists. Default is False.

    Returns
    -------
    pandas DataFrame
        The cell type edge list as a DataFrame. Consists of the columns 'source',
        'source_layer', 'target', 'target_layer', 'weight'.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"[ERROR] input_dir: {input_dir} is not a directory.")

    output_file = Path(output_file)
    if output_file.exists():
        if overwrite:
            print(f"[STATUS] Overwriting {output_file}.", flush=True)
        else:
            raise FileExistsError(f"[ERROR] output_file: {output_file} already exists.")

    # Generator of paths of (intra-)layer adjacency matrices
    layer_paths = input_dir.glob("*_layer_*.csv")

    # List of pandas DataFrames which represent the intra- and inter-layer edge lists
    edgelist_dataframes = []

    for layer in layer_paths:
        # Extract the name of the layer
        layer_name = layer.stem.split("_layer_")[-1]
        if layer_name == "celltype":
            print(
                f"[STATUS] Found adjacency matrix for layer {layer_name}.", flush=True
            )
            # Adjacency matrix as pandas DataFrame
            adj_df = pd.read_csv(layer, sep=input_sep, index_col=0)
            # Cast index as str (in case node names are integers)
            adj_df.index = adj_df.index.astype(str)
            # Turn adjacency matrix into NetworkX graph
            layer_nx = nx.from_pandas_adjacency(
                adj_df, create_using=nx.DiGraph if directed else nx.Graph
            )
            # Turn NetworkX graph into an edge list DataFrame
            layer_edgelist_df = nx.to_pandas_edgelist(layer_nx)
            # Append layer information
            layer_edgelist_df["source_layer"] = layer_name
            layer_edgelist_df["target_layer"] = layer_name

            # Add to list of intra-layer edge lists
            edgelist_dataframes.append(layer_edgelist_df)

    if len(edgelist_dataframes) < 1:
        raise FileNotFoundError(
            f"[ERROR] No cell type adjacency matrix of the form '*_layer_celltype.csv'"
            f" found in the directory {input_dir}."
        )

    # Concatenate all edge lists
    celltype_df = pd.concat(edgelist_dataframes, ignore_index=True)
    # Save multilayer edge list
    celltype_df.loc[
        :, ["source", "source_layer", "target", "target_layer", "weight"]
    ].to_csv(output_file, sep=output_sep, index=False, header=output_header)


def recursive_rename_adj_matrices(root_dir):
    """
    Recursively rename the myocardial infarct data adjacency matrices to match the
    naming scheme expected by `adjacency_matrices_to_multilayer_edgelist()`.

    Parameters
    ----------
    root_dir : str or pathlib.Path object
        Path to the root data directory.
    """
    root_dir = Path(root_dir)

    # Iterate over all sample directories
    for sample in root_dir.iterdir():
        if sample.is_dir():
            for csv_file in sample.iterdir():
                split = csv_file.name.split("_layer_")

                if split[-1] == "scRNAcellTypes.csv":
                    new_path = sample.joinpath(split[0] + "_layer_" + "celltype.csv")
                    csv_file.rename(new_path)

                elif split[-1] == "mapping.csv":
                    new_path = sample.joinpath(
                        split[0] + "_interlayer_" + "spot_celltype.csv"
                    )
                    csv_file.rename(new_path)

                elif split[-1] == "spot.csv":
                    continue

                else:
                    print(
                        f"[WARNING] Unrecognized file:"
                        f" {csv_file.relative_to(root_dir)}.",
                        flush=True,
                    )


def get_interlayer_weight_mean_and_std(adj_matrix_df):
    """
    Calculate the mean and standard deviation of the interlayer edge weights.

    Parameters
    ----------
    adj_matrix_df : pandas DataFrame
        The interlayer adjacency matrix.

    Returns
    -------
    mean : float
        Mean interlayer edge weight.
    std : float
        Standard deviation of the interlayer edge weights.
    """
    mean = adj_matrix_df.to_numpy().mean()
    std = adj_matrix_df.to_numpy().std(ddof=1)  # Delta degrees of freedom = 1

    return mean, std


def interlayer_k_neighbors(adj_matrix_df, neighbors, weights="similarity", axis=1):
    """
    Connect each node in layer1 with its k nearest neighbors in layer2.

    Parameters
    ----------
    adj_matrix_df : pandas DataFrame
        The interlayer adjacency matrix.
    neighbors : int
        The number of neighbors in layer2 to connect each node in layer1 to.
    weights : 'distance' or 'similarity'
        Whether edge weights represent distance or similarity between nodes. Default is
        `'similarity'`.
    axis : 0 (index) or 1 (columns)
        Specifies whether nodes in layer1 are the index or the columns of the DataFrame.
        (Nodes in layer1 get connected to their k nearest neighbors in layer2. Note that
        this is not symmetrical). Default is `1` (columns).

    Returns
    -------
    pandas DataFrame
        The new interlayer adjacency matrix with weights as floats.
    """
    # Transpose DataFrame if axis == 0
    if axis == 0:
        adj_matrix_df = adj_matrix_df.transpose()

    columns = adj_matrix_df.columns
    index = adj_matrix_df.index

    # If each node has k or less edges, return original adjacency matrix
    if neighbors >= len(index):
        return adj_matrix_df.transpose() if axis == 0 else adj_matrix_df

    # Dictionary for the new adjacency matrix
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

        # Set edge values in adjacency_dict for nearest neighbors of each node
        for i in range(neighbors):
            neighbor_index = sorting_indices.iloc[i]
            neighbor_label = column.index[neighbor_index]
            edge_value = column.iloc[neighbor_index]
            adjacency_dict[current_label][neighbor_label] = edge_value

        # Set edge values to zero for all other neighbors
        for i in range(neighbors, len(index)):
            neighbor_index = sorting_indices.iloc[i]
            neighbor_label = column.index[neighbor_index]
            adjacency_dict[current_label][neighbor_label] = 0

    # Create a new DataFrame
    new_adj_matrix = pd.DataFrame(adjacency_dict, dtype=float)

    # Transpose the DataFrame back to original orientation
    if axis == 0:
        new_adj_matrix = new_adj_matrix.transpose()
        # Sort columns
        new_adj_matrix.sort_index(axis="columns", inplace=True)
    else:
        new_adj_matrix.sort_index(axis="index", inplace=True)

    return new_adj_matrix


def interlayer_threshold(adj_matrix_df, threshold, remove, inplace=False):
    """
    Threshold interlayer edges. Set edge weights below/above threshold to 0.

    Parameters
    ----------
    adj_matrix_df : pandas DataFrame
        The interlayer adjacency matrix.
    threshold : float
        Threshold value.
    remove : "below" or "above"
        Whether to remove the edges below or above the threshold.
    inplace : bool
        Whether to perform the operation in place on the data. Default is `False`.

    Returns
    -------
    pandas DataFrame
        The new interlayer adjacency matrix with edges below/above threshold set to 0.
        None if `inplace=True`.
    """
    if not remove in ["below", "above"]:
        raise ValueError(
            f"[ERROR] remove needs to be one of 'below' or 'above' but is '{remove}'."
        )

    if remove == "below":
        new_matrix = adj_matrix_df.mask(
            adj_matrix_df < threshold, other=0, inplace=inplace
        )
    elif remove == "above":
        new_matrix = adj_matrix_df.mask(
            adj_matrix_df > threshold, other=0, inplace=inplace
        )

    return new_matrix


def sparsify_layers(
    root_dir,
    output_root,
    celltype_weights,
    celltype_k,
    spot_weights,
    spot_k,
    interlayer_weights,
    interlayer_method="threshold",
    interlayer_stds=None,
    interlayer_k=None,
    sep=",",
):
    """
    Sparsify the layers of the myocardial infarction networks. Spot and cell type
    layers are sparsified by computing k-nearest neighbor (KNN) graphs, and the
    interlayer edges are sparsified using a threshold value or by computing KNN.

    Parameters
    ----------
    root dir : str or pathlib.Path object
        Path to the root data directory.
    output_root : str or pathlib.Path object
        Path to the output directory where the KNN graphs will be saved.
    celltype_weights : "distance" or "similarity"
        Whether edge weights in the celltype layer represent distance or similarity
        between nodes.
    celltype_k : int or None
        Number of nearest neighbors to connect each node in the cell type layer to. If
        None, cell type layer will not be sparsified.
    spot_weights : "distance" or "similarity"
        Whether edge weights in the spot layer represent distance or similarity between
        nodes.
    spot_k : int or None
        Number of nearest neighbors to connect each node in the spot layer to. If None,
        spot layer will not be sparsified.
    interlayer_weights : "distance" or "similarity"
        Whether edge weights between layers represent distance or similarity between
        nodes.
    interlayer_method : "threshold", "knn", or None
        Method used for sparsifying the interlayer edges. If None, interlayer edges will
        not be sparsified. Default is `"threshold"`.
    interlayer_stds : int or None
        Number of standard deviations away from the mean where the threshold will be
        set for sparcifying interlayer edges. Only used if
        `interlayer_method="threshold". Default is None.
    interlayer_k : int or None
        Number of closest nodes in the spot layer which each node in the cell type layer
        shall be connected to. Only used if `interlayer_method="knn"`. Default is
        None.
    sep : str
        Delimiter of the input and output .csv files. Default is `","`.
    """
    root_dir = Path(root_dir)
    output_root = Path(output_root)

    if not output_root.exists():
        Path.mkdir(output_root)

    # Iterate over all samples
    for sample in root_dir.iterdir():
        if sample.is_dir():
            sample_name = sample.name.split("_MLN")[0]
            print(f"[STATUS] Processing sample {sample_name}.", flush=True)
            output_dir = output_root.joinpath(sample_name)
            Path.mkdir(output_dir)

            for csv_file in sample.iterdir():
                # Cell type layer
                if "_layer_celltype" in csv_file.name:
                    if celltype_k is not None:
                        print(
                            f"[STATUS] Sparsifying cell type layer of sample {sample_name}.",
                            flush=True,
                        )
                        adj_matrix_df = pd.read_csv(csv_file, index_col=0, sep=sep)
                        knn_df = knn_graph.knn_graph_from_adjacency_matrix(
                            adj_matrix_df, k=celltype_k, weights=celltype_weights
                        )
                        output_file = output_dir.joinpath(
                            sample_name + f"_{celltype_k}NN_layer_celltype.csv"
                        )
                        knn_df.to_csv(output_file, sep=sep)
                    else:
                        # If not to be sparsified, copy original file
                        shutil.copy(csv_file, output_dir)

                # Spot layer
                elif "_layer_spot" in csv_file.name:
                    if spot_k is not None:
                        print(
                            f"[STATUS] Sparsifying spot layer of sample {sample_name}.",
                            flush=True,
                        )
                        adj_matrix_df = pd.read_csv(csv_file, index_col=0, sep=sep)

                        # Cast header as ints
                        adj_matrix_df.columns = adj_matrix_df.columns.astype(int)

                        knn_df = knn_graph.knn_graph_from_adjacency_matrix(
                            adj_matrix_df, k=spot_k, weights=spot_weights
                        )
                        output_file = output_dir.joinpath(
                            sample_name + f"_{spot_k}NN_layer_spot.csv"
                        )
                        knn_df.to_csv(output_file, sep=sep)
                    else:
                        # If not to be sparsified, copy original file
                        shutil.copy(csv_file, output_dir)

                # Interlayer adjacency matrix
                elif "_interlayer_spot_celltype" in csv_file.name:
                    if interlayer_method is not None:
                        print(
                            f"[STATUS] Sparsifying interlayer connections of sample"
                            f" {sample_name}.",
                            flush=True,
                        )
                        adj_matrix_df = pd.read_csv(csv_file, index_col=0, sep=sep)

                        if interlayer_method == "knn":
                            if type(interlayer_k) != int:
                                raise ValueError(
                                    f"[ERROR] interlayer_k must be of type int when interla"
                                    f"yer_method='knn', but is of type {type(interlayer_k)}."
                                )
                            knn_df = interlayer_k_neighbors(
                                adj_matrix_df,
                                neighbors=interlayer_k,
                                weights=interlayer_weights,
                                axis=1,
                            )
                            output_file = output_dir.joinpath(
                                sample_name
                                + f"_{interlayer_k}NN_interlayer_spot_celltype.csv"
                            )
                            knn_df.to_csv(output_file, sep=sep)

                        elif interlayer_method == "threshold":
                            # Calculate interlayer edge weight mean and std
                            mean, std = get_interlayer_weight_mean_and_std(
                                adj_matrix_df
                            )
                            # If weights are "distance", remove all weights above
                            # threshold, otherwise remove all weights below threshold.
                            if interlayer_weights == "distance":
                                threshold_value = mean - (interlayer_stds * std)
                                remove = "above"
                            elif interlayer_weights == "similarity":
                                threshold_value = mean + (interlayer_stds * std)
                                remove = "below"
                            else:
                                raise ValueError(
                                    f"[ERROR] interlayer_weights should be 'distance' or"
                                    f" 'similarity' but is '{interlayer_weights}'."
                                )
                            thresh_df = interlayer_threshold(
                                adj_matrix_df,
                                threshold_value,
                                remove,
                                inplace=False,
                            )
                            output_file = output_dir.joinpath(
                                sample_name + f"_thresh_interlayer_spot_celltype.csv"
                            )
                            thresh_df.to_csv(output_file, sep=sep)
                        else:
                            raise ValueError(
                                f"[ERROR] interlayer_method should be 'knn' or 'threshold'"
                                f" but is '{interlayer_method}'."
                            )
                    else:
                        # If not to be sparsified, copy original file
                        shutil.copy(csv_file, output_dir)


def invert_weights_adjacency_matrix(
    in_file,
    sep=",",
    save_file=False,
    output_file=None,
    overwrite=False,
    delete_input_file=False,
):
    """
    Read the .csv file of an adjacency matrix and invert all weights (1/weight). Zero
    entries remain zero.

    Parameters
    ----------
    in_file : str or pathlib.Path object
        Path of adjacency matrix .csv file.
    sep : str
        Delimiter for the input and output .csv file. Default is `","`.
    save_file : bool
        Whether to save the resulting adjacency matrix as a .csv file. Default is
        `False`.
    output_file : str or pathlib.Path object
        Path of the output .csv file. If `out_file` is a directory, the output file will
        be saved with the same name as the input file. Default is None.
    overwrite : bool
        Whether to overwrite existing .csv file. Default is `False`.
    delete_input_file : bool
        Whether to delete the input .csv after inverting the weights and saving the new
        .csv file. Only applies if `save_file=True`. Default is `False`.

    Returns
    -------
    pandas DataFrame
        The adjacency matrix with inverted weights as a DataFrame.
    """
    in_file = Path(in_file)
    if not in_file.exists():
        raise FileNotFoundError(f"[ERROR] Input file {in_file} does not exist.")

    input_df = pd.read_csv(in_file, sep=sep, index_col=0)
    inverse_df = 1 / input_df
    inverse_df.replace(np.inf, 0.0, inplace=True)

    if save_file:
        if not output_file:
            raise ValueError(
                f"[ERROR] output_file cannot be None if save_file is True."
            )
        out_path = Path(output_file)
        if out_path.is_dir():
            out_path = out_path.joinpath(in_file.name)
        if out_path.is_file() and not overwrite:
            raise FileExistsError(f"[ERROR] File {out_path} already exists.")

        if not out_path.exists():
            print(f"[STATUS] Created file {out_path}.", flush=True)
        if out_path.suffix == ".csv":
            inverse_df.to_csv(out_path, sep=sep)
        else:
            raise ValueError(f"[ERROR] output_file: {output_file} is not a .csv file.")

        if delete_input_file:
            print(f"[STATUS] Deleted file {in_file}.", flush=True)
            in_file.unlink()

    return inverse_df


def min_max_normalization_by_column(dataframe, range=[0, 1]):
    """
    Normalize the values of a DataFrame by column using min-max normalization. Each
    column is normalized individually.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The DataFrame to normalize.
    range : `[a,b]` where `a` and `b` are ints
        The range to normalize the data to. The minimum value of each column will take
        on the value `a` and the maximum value will take on the value `b`. Default is
        `[0,1]`.

    Returns
    -------
    pandas DataFrame
        The normalized DataFrame.
    """
    df = dataframe
    a = range[0]
    b = range[1]

    for column in df.columns:
        min = df[column].min()
        max = df[column].max()
        if min == max:
            df[column] = float(a)
        else:
            df[column] = df[column].sub(min)
            df[column] = df[column].mul(float(b - a) / (max - min))
            df[column] = df[column].add(a)

    return df


def min_max_normalization(dataframe, range=[0, 1]):
    """
    Normalize the values of a DataFrame using min-max normalization. The values across
    the whole DataFrame are normalized.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The DataFrame to normalize.
    range : `[a,b]` where `a` and `b` are ints
        The range to normalize the data to. The minimum value in the DataFrame will take
        on the value `a` and the maximum value will take on the value `b`. Default is
        `[0,1]`.

    Returns
    -------
    pandas DataFrame
        The normalized DataFrame.
    """
    df = dataframe
    a = range[0]
    b = range[1]

    min = df.min(axis=None)
    max = df.max(axis=None)
    if min == max:
        df = float(a)
    else:
        df = df.sub(min)
        df = df.mul(float(b - a) / (max - min))
        df = df.add(a)

    return df
