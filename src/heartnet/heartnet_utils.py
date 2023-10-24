from pathlib import Path
import pandas as pd
import tomli
import math


def load_heartnet_config(config_path):
    """
    Load configuration file and return config as dict.

    Parameters
    ----------
    config_path : str or pathlib.Path object
        Path of the config .toml file.

    Returns
    -------
    dict
        Config as a dict.
    """
    config_path = Path(config_path)

    if not (config_path.exists() and config_path.suffix == ".toml"):
        raise ValueError(
            f"[ERROR] config_path {config_path} is not a valid path of a .toml file."
        )

    with open(config_path, mode="rb") as f:
        config = tomli.load(f)

    return config


def try_dict_value(dictionary, key, default):
    """
    Try loading the value at dictionary[key] if it exists. If it doesn't, return
    default.

    Parameters
    ----------
    dictionary : dict
    key : immutable datatype
    default : Any

    Returns
    -------
    dictionary[key] if it exists, else default.
    """
    try:
        return dictionary[key]
    except KeyError:
        return default


def config_file_to_flat_dict(config_path):
    """
    Load configuration file and return config as dict containing the relevant
    configuration parameters as keys.

    Parameters
    ----------
    config_path : str or pathlib.Path object
        Path of the config .toml file.

    Returns
    -------
    dict
        Config as a dict with the following keys:
            spot_k : int (NaN if spot layer is not sparsified)
            celltype_k : int (NaN if celltype layer is not sparsified)
            interlayer_k : int (NaN if interlayer edges not sparsified as knn)
            interlayer_stds : int (NaN if interlayer edges not sparsified using threshold)

            spot_norm : 'norm' or 'colnorm'
            celltype_norm : 'norm' or 'colnorm'
            interlayer_norm : 'norm' or 'colnorm'

            p : float
            q : float
            s_c2s : float (s from celltype to spot layer)
            s_s2c : float (s from spot to celltype layer)
            dims : int
            num_walks : int
            walk_length : int
            window_size : int
            epochs : int

            epsilon : float (0 if not regularized)
            distribution : 'uniform' or 'balanced'
            alpha : float (NaN if distribution is 'balanced')
    """
    config = load_heartnet_config(config_path)
    flat = {}

    sparsification = config["preprocessing"]["sparsification"]
    if sparsification["sparsify_spot"]:
        flat["spot_k"] = int(sparsification["spot_k"])
    else:
        flat["spot_k"] = math.nan

    if sparsification["sparsify_celltype"]:
        flat["celltype_k"] = int(sparsification["celltype_k"])
    else:
        flat["celltype_k"] = math.nan

    if sparsification["sparsify_interlayer"]:
        if sparsification["interlayer_method"] == "knn":
            flat["interlayer_k"] = int(sparsification["interlayer_k"])
            flat["interlayer_stds"] = math.nan
        elif sparsification["interlayer_method"] == "threshold":
            flat["interlayer_k"] = math.nan
            flat["interlayer_stds"] = try_dict_value(
                sparsification, "interlayer_stds", 1
            )
    else:
        flat["interlayer_k"] = math.nan
        flat["interlayer_stds"] = math.nan

    normalization = config["preprocessing"]["normalization"]
    flat["spot_norm"] = normalization["spot"]
    flat["celltype_norm"] = normalization["celltype"]
    flat["interlayer_norm"] = normalization["interlayer"]

    embedding = config["embedding"]["parameters"]
    flat["p"] = round(embedding["p"], 1)
    flat["q"] = round(embedding["q"], 1)
    if type(embedding["s"]) in [int, float]:
        flat["s_c2s"] = embedding["s"]
        flat["s_s2c"] = embedding["s"]
    else:
        for d in embedding["s"]:
            if d["source"] == "celltype" and d["target"] == "spot":
                flat["s_c2s"] = d["s"]
            elif d["source"] == "spot" and d["target"] == "celltype":
                flat["s_s2c"] = d["s"]

    flat["dims"] = int(embedding["dims"])
    flat["num_walks"] = int(embedding["num_walks"])
    flat["walk_length"] = int(embedding["walk_length"])
    flat["window_size"] = int(embedding["window_size"])
    flat["epochs"] = int(embedding["epochs"])

    gw = config["gromov_wasserstein"]
    if gw["calculate_entr_gw"]:
        flat["epsilon"] = gw["epsilon"]
    else:
        flat["epsilon"] = 0

    flat["distribution"] = gw["distribution"]
    if gw["distribution"] == "balanced":
        flat["alpha"] = gw["alpha"]
    else:
        flat["alpha"] = math.nan

    return flat


def config_files_to_dataframe(config_directory):
    """
    Load all config files in config_directory and return the config values in one Pandas
    DataFrame. The returned DataFrame has the config parameters as columns and a row
    for each config file.

    Parameters
    ----------
    config_directory : str or pathlib.Path object
        Path of the directory containing all config .toml files.

    Returns
    -------
    Pandas DataFrame
        DataFrame containing the parameters of all config files. Config parameters are
        columns and each row is a separate config.
    """
    config_dir = Path(config_directory)

    if not (config_dir.exists() and config_dir.is_dir()):
        raise ValueError(f"[ERROR] {config_dir} is not a valid directory.")

    # List to collect all config dictionaries
    data = []

    for config_file in config_dir.iterdir():
        if config_file.is_file() and config_file.suffix == ".toml":
            if not config_file.name.startswith("config"):
                print(
                    f"[WARNING] Skipping file {config_file.name} because it's name does"
                    f" not start with 'config'."
                )

            config_num = int(config_file.stem.split("config")[1])
            config_dict = config_file_to_flat_dict(config_file)
            config_dict["config num"] = config_num
            data.append(config_dict)

    return pd.DataFrame(data).set_index("config num").sort_index()


def label_csv_to_dict(csv_path):
    """
    Turn a .csv file containing the mapping of samples to labels into a dict of form
    {sample_name : label}. The input .csv file contains two columns: sample name and
    label.

    Parameters
    ----------
    csv_path : str or pathlib.Path object
        Path of the label .csv file.

    Returns
    -------
    dict
        Dictionary of the form {sample_name : true_label}.
    """
    csv_path = Path(csv_path)
    label_df = pd.read_csv(csv_path)
    label_dict = {}

    for index, row in label_df.iterrows():
        label_dict[row["patient_region_id"]] = row["patient_group"]

    return label_dict


def assert_valid_config(config):
    """
    Assert that all config entries and values are valid. Raises error if config is
    invalid.

    Parameters
    ----------
    config : dict
        Config dict.
    """
    # Assert presence of top-level config tables
    top_level_tables = [
        "data",
        "preprocessing",
        "embedding",
        "gromov_wasserstein",
        "clustering",
    ]
    for t in top_level_tables:
        assert t in config, f"Invalid config. Config must contain [{t}] table."

    assert (
        type(config["processes"]) == int and config["processes"] > 0
    ), f"Invalid config. [processes] must be an Integer greater than 0."

    # ----------------------------------------------------------------------------------
    # DATA CONFIG
    # ----------------------------------------------------------------------------------
    data = config["data"]
    for key in data:
        assert (
            type(key) == str and Path(data[key]).exists()
        ), f"Invalid config. [data.{key}] must be a valid path."

    # ----------------------------------------------------------------------------------
    # PREPROCESSING CONFIG
    # ----------------------------------------------------------------------------------
    preprocessing = config["preprocessing"]
    assert preprocessing["weights"] in ["distance", "similarity"], (
        f"Invalid config. 'weights' must be 'distance' or 'similarity' but is"
        f" '{preprocessing['weights']}'."
    )

    # Sparsification
    sparsification = preprocessing["sparsification"]

    assert isinstance(
        sparsification["sparsify_spot"], bool
    ), "Invalid config. 'sparsify_spot' must be a Boolean."
    assert isinstance(
        sparsification["sparsify_celltype"], bool
    ), "Invalid config. 'sparsify_celltype' must be a Boolean."
    assert isinstance(
        sparsification["sparsify_interlayer"], bool
    ), "Invalid config. 'sparsify_interlayer' must be a Boolean."

    if sparsification["sparsify_spot"]:
        assert isinstance(
            sparsification["spot_k"], int
        ), "Invalid config. 'spot_k' must be an Integer."
    if sparsification["sparsify_celltype"]:
        assert isinstance(
            sparsification["celltype_k"], int
        ), "Invalid config. 'celltype_k' must be an Integer."
    if sparsification["sparsify_interlayer"]:
        assert sparsification["interlayer_method"] in ["threshold", "knn"], (
            f"Invalid config. 'interlayer_method' must be 'threshold' or 'knn' but is"
            f" '{sparsification['interlayer_method']}'."
        )
        if sparsification["interlayer_method"] == "threshold":
            assert type(sparsification["interlayer_stds"]) in [
                int,
                float,
            ], "Invalid config. 'interlayer_stds' must be an Integer or a Float."
        if sparsification["interlayer_method"] == "knn":
            assert isinstance(
                sparsification["interlayer_k"], int
            ), "Invalid config. 'interlayer_k' must be an Integer."

    # Normalization
    normalization = preprocessing["normalization"]
    assert (
        type(normalization["spot"]) == bool
    ), f"Invalid config. [normalization.spot] must be a Boolean."
    assert (
        type(normalization["celltype"]) == bool
    ), f"Invalid config. [normalization.celltype] must be a Boolean."
    assert (
        type(normalization["interlayer"]) == bool
    ), f"Invalid config. [normalization.interlayer] must be a Boolean."

    # ----------------------------------------------------------------------------------
    # EMBEDDING CONFIG
    # ----------------------------------------------------------------------------------
    embedding = config["embedding"]
    assert embedding["algorithm"] in ["HeNHoE-2vec"], (
        f"Invalid config. [embedding.algorithm] must be 'HeNHoE-2vec' but is"
        f" '{embedding['algorithm']}'."
    )

    # Embedding parameters
    parameters = embedding["parameters"]
    assert (
        type(parameters["p"]) in [int, float] and parameters["p"] > 0
    ), "Invalid config. 'p' must be an Integer or a Float greater than 0."
    assert (
        type(parameters["q"]) in [int, float] and parameters["q"] > 0
    ), "Invalid config. 'q' must be an Integer or a Float greater than 0."

    assert type(parameters["s"]) in [
        int,
        float,
        list,
    ], "Invalid config. 's' must be an Integer, Float, or Array."
    if type(parameters["s"]) in [int, float]:
        assert parameters["s"] > 0, "Invalid config. 's' must be greater than 0."
    else:
        for d in parameters["s"]:
            assert (
                type(d) == dict
            ), "Invalid config. Every element of 's' must be a table."
            keys = ["source", "target", "s"]
            for key in keys:
                assert (
                    key in d
                ), f"Invalid config. Every element of 's' must contain the key '{key}'."
            assert d["s"] > 0, (
                "Invalid config. Every element of 's' must contain an entry 's' which"
                " is greater than 0."
            )

    assert (
        type(parameters["dims"]) == int and parameters["dims"] > 0
    ), "Invalid config. 'dims' must be an Integer greater than 0."
    assert (
        type(parameters["num_walks"]) == int and parameters["num_walks"] > 0
    ), "Invalid config. 'num_walks' must be an Integer greater than 0."
    assert (
        type(parameters["walk_length"]) == int and parameters["walk_length"] > 0
    ), "Invalid config. 'walk_length' must be an Integer greater than 0."
    assert (
        type(parameters["window_size"]) == int
        and parameters["window_size"] > 0
        and parameters["window_size"] <= parameters["walk_length"]
    ), (
        "Invalid config. 'window_size' must be an Integer greater than 0 and smaller"
        " than 'walk_length'."
    )
    assert (
        type(parameters["epochs"]) == int and parameters["epochs"] > 0
    ), "Invalid config. 'epochs' must be an Integer greater than 0."

    # ----------------------------------------------------------------------------------
    # GROMOV-WASSERSTEIN CONFIG
    # ----------------------------------------------------------------------------------
    gw = config["gromov_wasserstein"]

    assert (
        type(gw["calculate_gw"]) == bool
    ), f"Invalid config. [gromov_wasserstein.calculate_gw] must be a Boolean."
    assert (
        type(gw["calculate_entr_gw"]) == bool
    ), f"Invalid config. [gromov_wasserstein.calculate_entr_gw] must be a Boolean."
    if gw["calculate_entr_gw"]:
        assert type(gw["epsilon"]) in [int, float] and gw["epsilon"] > 0, (
            f"Invalid config. [gromov_wasserstein.epsilon] must be a float greater than"
            f" 0."
        )
    assert gw["distribution"] in ["balanced", "uniform"], (
        f"Invalid config. [gromov_wasserstein.distribution] must be 'balanced' or"
        f" 'uniform' but is '{gw['distribution']}'."
    )
    if gw["distribution"] == "balanced":
        assert (
            gw["alpha"] >= 0 and gw["alpha"] <= 1
        ), "Invalid config. [gromov_wasserstein.alpha] must be a Float between 0 and 1."

    # ----------------------------------------------------------------------------------
    # CLUSTERING CONFIG
    # ----------------------------------------------------------------------------------
    clustering = config["clustering"]

    assert clustering["linkage_method"] in ["single", "complete", "average", "ward"], (
        f"Invalid config. [clustering.linkage_method] must be one of 'single',"
        f" 'complete', 'average', or 'ward' but is '{clustering['linkage_method']}'."
    )
    assert (
        type(clustering["n_clusters"]) == int and clustering["n_clusters"] > 0
    ), f"Invalid config. [clustering.n_clusters] must be an Integer greater than 0."
