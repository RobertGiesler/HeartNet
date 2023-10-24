from . import heartnet_utils
from . import data_preprocessing
from . import gromov_wasserstein
from . import gw_utils
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hierarchy
import time
import argparse
import henhoe2vec as hh2v


def main(config_path):
    # ----------------------------------------------------------------------------------
    # LOAD AND ASSERT CONFIG
    # ----------------------------------------------------------------------------------
    config_path = Path(config_path)
    config = heartnet_utils.load_heartnet_config(config_path)
    heartnet_utils.assert_valid_config(config)

    # ----------------------------------------------------------------------------------
    # PREPROCESS DATA
    # ----------------------------------------------------------------------------------
    config_name = config_path.stem
    start = time.time()
    print(f"[STATUS] Starting HeartNet with config '{config_name}'.", flush=True)

    raw_data = Path(config["data"]["raw_data"])
    preprocessed_data = Path(config["data"]["preprocessed_data"])
    edgelists = Path((config["data"]["edgelists"]))

    print(f"[STATUS] Preprocessing data...", flush=True)
    # Prepare name of preprocessed data
    name = "spot"
    sparsification = config["preprocessing"]["sparsification"]
    normalization = config["preprocessing"]["normalization"]
    if sparsification["sparsify_spot"]:
        name += f"_{sparsification['spot_k']}NN"
    if normalization["spot"]:
        name += f"_norm"
    name += "_celltype"
    if sparsification["sparsify_celltype"]:
        name += f"_{sparsification['celltype_k']}NN"
    if normalization["celltype"]:
        name += f"_norm"
    name += "_inter"
    if sparsification["sparsify_interlayer"]:
        if sparsification["interlayer_method"] == "threshold":
            name += f"_{sparsification['interlayer_stds']}thresh"
        else:
            name += f"_{sparsification['interlayer_k']}NN"
    if normalization["interlayer"]:
        name += f"_norm"

    # Prepare preprocessed data directory
    prepro_data_dir = preprocessed_data.joinpath(name)
    if prepro_data_dir.exists():
        print(
            f"[STATUS] Using preprocessed data found at {prepro_data_dir}.", flush=True
        )
    else:
        # Sparsify
        print(f"[STATUS] Sparsifying layers...", flush=True)
        # Prepare parameters for sparsification
        if not sparsification["sparsify_spot"]:
            spot_k = None
        else:
            spot_k = sparsification["spot_k"]

        if not sparsification["sparsify_celltype"]:
            celltype_k = None
        else:
            celltype_k = sparsification["celltype_k"]

        if not sparsification["sparsify_interlayer"]:
            interlayer_method = None
            interlayer_k = None
        elif sparsification["interlayer_method"] == "threshold":
            interlayer_method = sparsification["interlayer_method"]
            interlayer_stds = sparsification["interlayer_stds"]
            interlayer_k = None
        else:
            interlayer_method = sparsification["interlayer_method"]
            interlayer_stds = None
            interlayer_k = sparsification["interlayer_k"]

        data_preprocessing.sparsify_layers(
            root_dir=raw_data,
            output_root=prepro_data_dir,
            celltype_weights="distance",
            celltype_k=celltype_k,
            spot_weights="distance",
            spot_k=spot_k,
            interlayer_weights="similarity",
            interlayer_method=interlayer_method,
            interlayer_stds=interlayer_stds,
            interlayer_k=interlayer_k,
            sep=",",
        )

        # Invert interlayer edge weights
        print(f"[STATUS] Inverting interlayer edge weights...", flush=True)
        for sample in prepro_data_dir.iterdir():
            if sample.is_dir():
                for csv_file in sample.iterdir():
                    if "_interlayer_spot_celltype" in csv_file.name:
                        split = csv_file.name.split("_interlayer_spot_celltype")
                        out_file = sample.joinpath(
                            split[0] + "_dist_interlayer_spot_celltype.csv"
                        )
                        data_preprocessing.invert_weights_adjacency_matrix(
                            csv_file,
                            save_file=True,
                            output_file=out_file,
                            overwrite=False,
                            delete_input_file=True,
                        )

        # Normalize
        print(f"[STATUS] Normalizing edge weights...", flush=True)
        for sample in prepro_data_dir.iterdir():
            if sample.is_dir():
                print(f"[STATUS] Normalizing sample {sample.name}.", flush=True)
                # Iterate over all adjacency matrices
                for csv_file in sample.iterdir():
                    # Interlayer edges
                    if "_dist_interlayer_spot_celltype" in csv_file.name:
                        normalize = normalization["interlayer"]
                        if normalize:
                            split = csv_file.name.split(
                                "_dist_interlayer_spot_celltype"
                            )
                            input_df = pd.read_csv(csv_file, index_col=0, sep=",")
                            normalized_df = data_preprocessing.min_max_normalization(
                                input_df
                            )
                            normalized_df.to_csv(csv_file, sep=",")
                            new_file = csv_file.parent.joinpath(
                                split[0] + f"_norm_dist_interlayer_spot_celltype.csv"
                            )
                            csv_file.rename(new_file)

                    # Cell type layer
                    elif "_layer_celltype" in csv_file.name:
                        normalize = normalization["celltype"]
                        if normalize:
                            split = csv_file.name.split("_layer_celltype")
                            input_df = pd.read_csv(csv_file, index_col=0, sep=",")
                            normalized_df = data_preprocessing.min_max_normalization(
                                input_df
                            )
                            normalized_df.to_csv(csv_file, sep=",")
                            new_file = csv_file.parent.joinpath(
                                split[0] + f"_norm_layer_celltype.csv"
                            )
                            csv_file.rename(new_file)

                    # Spot layer
                    elif "_layer_spot" in csv_file.name:
                        normalize = normalization["spot"]
                        if normalize:
                            split = csv_file.name.split("_layer_spot")
                            input_df = pd.read_csv(csv_file, index_col=0, sep=",")
                            normalized_df = data_preprocessing.min_max_normalization(
                                input_df
                            )
                            normalized_df.to_csv(csv_file, sep=",")
                            new_file = csv_file.parent.joinpath(
                                split[0] + f"_norm_layer_spot.csv"
                            )
                            csv_file.rename(new_file)

    # Turn adjacency matrices into multilayer edgelists
    print(f"[STATUS] Converting adjacency matrices into edgelists...", flush=True)
    edgelists_dir = edgelists.joinpath(name)
    # Check if edgelists already exist
    if edgelists_dir.exists():
        print(f"[STATUS] Using edgelists found at {edgelists_dir}.", flush=True)
    # Convert adjacency matrices to edgelists
    else:
        Path.mkdir(edgelists_dir)
        for sample in prepro_data_dir.iterdir():
            if sample.is_dir():
                sample_name = sample.name
                print(
                    f"[STATUS] Converting adjacency matrices of sample {sample_name}.",
                    flush=True,
                )
                output_file = edgelists_dir.joinpath(sample_name + ".edg")

                data_preprocessing.adjacency_matrices_to_multilayer_edgelist(
                    sample, output_file, output_header=True
                )

    end_prepro = time.time()
    seconds = round((end_prepro - start), 1)
    minutes = int(seconds // 60)
    rem_seconds = round(seconds % 60, 1)
    hours = int(minutes // 60)
    rem_minutes = round(minutes % 60, 1)
    print(
        f"[STATUS] Finished preprocessing for config '{config_name}' after"
        f" {hours} hours {rem_minutes} mins {rem_seconds} seconds.",
        flush=True,
    )

    # ----------------------------------------------------------------------------------
    # NODE EMBEDDING
    # ----------------------------------------------------------------------------------
    start_emb = time.time()
    print(f"[STATUS] Computing node embeddings...", flush=True)
    # Prepare embeddings directory
    date_time = datetime.today().strftime("%Y-%m-%dT%H%M%S")
    embeddings_dir = Path(config["data"]["embeddings"]).joinpath(
        f"{config_name}_{date_time}"
    )
    Path.mkdir(embeddings_dir)

    embedding = config["embedding"]
    emb_params = embedding["parameters"]

    if embedding["algorithm"] == "HeNHoE-2vec":
        p = emb_params["p"]
        q = emb_params["q"]
        s = emb_params["s"]

        # If s is a list, convert into dict
        if type(s) == list:
            s_dict = {}
            for layer_pair in s:
                layer_tuple = (layer_pair["source"], layer_pair["target"])
                s_dict[layer_tuple] = layer_pair["s"]
            s = s_dict

        dims = emb_params["dims"]
        num_walks = emb_params["num_walks"]
        walk_length = emb_params["walk_length"]
        window_size = emb_params["window_size"]
        epochs = emb_params["epochs"]
        workers = config["processes"]

        for edgelist in edgelists_dir.iterdir():
            if ".edg" in edgelist.name:
                print(
                    f"[STATUS] Computing node embeddings of sample {edgelist.stem}.",
                    flush=True,
                )
                output_name = edgelist.stem + "_embeddings"
                hh2v.henhoe2vec.run(
                    edgelist,
                    embeddings_dir,
                    sep=",",
                    header=True,
                    output_name=output_name,
                    edges_are_distance=True,
                    p=p,
                    q=q,
                    s=s,
                    dims=dims,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    window_size=window_size,
                    epochs=epochs,
                    workers=workers,
                )

    end_emb = time.time()
    seconds = round((end_emb - start_emb), 1)
    minutes = int(seconds // 60)
    rem_seconds = round(seconds % 60, 1)
    hours = int(minutes // 60)
    rem_minutes = round(minutes % 60, 1)
    print(
        f"[STATUS] Finished all node embeddings for config '{config_name}' after"
        f" {hours} hours {rem_minutes} mins {rem_seconds} seconds.",
        flush=True,
    )

    # ----------------------------------------------------------------------------------
    # CALCULATE GROMOV-WASSERSTEIN DISTANCES
    # ----------------------------------------------------------------------------------
    print(f"[STATUS] Calculating Gromov-Wasserstein distances...", flush=True)
    gw_config = config["gromov_wasserstein"]
    # Prepare output directory
    gw_dir = Path(config["data"]["gw_dir"]).joinpath(f"{config_name}_{date_time}")

    gromov_wasserstein.gw_embeddings(
        embeddings_dir=embeddings_dir,
        output_dir=gw_dir,
        distribution=gw_config["distribution"],
        alpha=gw_config["alpha"],
        calculate_gw=gw_config["calculate_gw"],
        calculate_entr_gw=gw_config["calculate_entr_gw"],
        epsilon=gw_config["epsilon"],
        processes=config["processes"],
    )

    # ----------------------------------------------------------------------------------
    # AGGLOMERATIVE HIERARCHICAL CLUSTERING
    # ----------------------------------------------------------------------------------
    print(f"[STATUS] Computing clusterings...", flush=True)
    start_clust = time.time()
    # Prepare clustering directory
    clusterings_dir = Path(config["data"]["clusterings"]).joinpath(
        f"{config_name}_{date_time}"
    )
    Path.mkdir(clusterings_dir)

    # Get GW distances
    gw_dist_dict = gw_utils.gw_dist_from_pkl(gw_dir)
    entr_gw_dist_dict = gw_utils.entr_gw_dist_from_pkl(gw_dir)

    # Turn into np.array
    gw_dist_array, index = gw_utils.tuple_dict_to_2d_array(gw_dist_dict, symmetric=True)
    entr_gw_dist_array, entr_index = gw_utils.tuple_dict_to_2d_array(
        entr_gw_dist_dict, symmetric=True
    )

    # Convert distance matrices into condensed form
    cond_dist_mat = squareform(gw_dist_array, checks=False)
    entr_cond_dist_mat = squareform(entr_gw_dist_array, checks=False)

    # Perform hierarchical clustering
    linkage_method = config["clustering"]["linkage_method"]
    n_clusters = config["clustering"]["n_clusters"]

    linkage = hierarchy.linkage(
        cond_dist_mat, optimal_ordering=True, method=linkage_method
    )
    entr_linkage = hierarchy.linkage(
        entr_cond_dist_mat, optimal_ordering=True, method=linkage_method
    )

    cluster_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters)
    entr_cluster_labels = hierarchy.cut_tree(entr_linkage, n_clusters=n_clusters)

    # Save predicted clusterings as DataFrames
    cluster_df = pd.DataFrame(
        data=cluster_labels, columns=["pred_cluster"], index=index
    )
    entr_cluster_df = pd.DataFrame(
        data=entr_cluster_labels, columns=["pred_cluster"], index=entr_index
    )

    # Save predicted clusterings
    cluster_df.to_csv(clusterings_dir.joinpath("clustering.csv"), sep="\t")
    entr_cluster_df.to_csv(clusterings_dir.joinpath("entr_clustering.csv"), sep="\t")

    end_clust = time.time()
    seconds = round(end_clust - start_clust, 5)
    print(
        f"[STATUS] Finished clustering for config '{config_name}' after {seconds}"
        f" seconds.",
        flush=True,
    )

    # ----------------------------------------------------------------------------------
    # END OF SCRIPT
    # ----------------------------------------------------------------------------------
    end = time.time()
    minutes = round((end - start) / 60, 1)
    hours = int(minutes // 60)
    rem_minutes = round(minutes % 60, 1)
    print(
        f"[STATUS] Finished HeartNet calculations for config '{config_name}' after"
        f" {hours} hours {rem_minutes} mins. See clustering results in"
        f" {clusterings_dir}.",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# RUNNING THE SCRIPT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run data preprocessing, network embedding, and Gromov-Wasserstein"
            " distance calculation, as specified by the config file."
        )
    )

    parser.add_argument(
        "--config", type=str, help="Path of the config .toml file to be used."
    )

    args = parser.parse_args()

    # Run main
    main(args.config)
