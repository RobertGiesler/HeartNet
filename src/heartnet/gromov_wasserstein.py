from pathlib import Path
import pandas as pd
import scipy as sp
import numpy as np
import ot
import time
import argparse
from multiprocessing import Pool
import pickle
import random


def gw_pairs(
    distance_matrices,
    distributions,
    sample_pairs,
    output_dir,
    calculate_gw=True,
    calculate_entr_gw=True,
    epsilon=5e-4,
):
    """
    Calculate the (entropic) Gromov-Wasserstein transport plans and distances between
    pairs of samples.

    Parameters
    ----------
    distance_matrices : dict of numpy.ndarray
        Dict of distance matrices of all samples: {sample_name : distance_matrix}.
    distributions : dict of numpy.ndarray
        Dict of probability distributions to use for GW mapping:
        {sample_name : distribution}.
    sample_pairs : list of 2-tuples of str
        List of sample pairs for which to calculate the GW transport plans and
        distances. E.g., [("RZ_P3","RZ_FZ_P5"),("FZ_P14","RZ_P9")].
    output_dir : str or pathlib.Path object
        Path of the output directory where the results will be saved.
    calculate_gw : bool
        Whether to calculate the non-entropic GW transport plans and distances. Default
        is True.
    calculate_entr_gw : bool
        Whether to calculate the entropic GW transport plans and distances. Default is
        True.
    epsilon : float
        The epsilon value to be used for entropic GW. Default is 5e-4.
    """
    # Prepare output directory
    output_root = Path(output_dir)
    if not output_root.exists():
        Path.mkdir(output_root)

    for sample_1, sample_2 in sample_pairs:
        # Prepare output directory
        output_dir = output_root.joinpath(f"GW_{sample_1}_{sample_2}")
        if not output_dir.exists():
            Path.mkdir(output_dir)

        # Extract distance matrices
        d1 = distance_matrices[sample_1]
        d2 = distance_matrices[sample_2]

        # Get probability distributions of samples
        p = distributions[sample_1]
        q = distributions[sample_2]

        # Calculate GW transport and distance
        if calculate_gw:
            print(
                f"[STATUS] Calculating GW between {sample_1} and {sample_2}.",
                flush=True,
            )
            start = time.time()
            gw, log = ot.gromov.gromov_wasserstein(
                d1, d2, p, q, "square_loss", verbose=False, log=True, max_iter=10000
            )

            # Save results
            output_plan = output_dir.joinpath(f"gw_plan_{sample_1}_{sample_2}.pkl")
            with open(output_plan, "wb") as f:
                pickle.dump(gw, f)

            output_log = output_dir.joinpath(f"gw_log_{sample_1}_{sample_2}.pkl")
            with open(output_log, "wb") as f:
                pickle.dump(log, f)

            end = time.time()
            print(
                f"[STATUS] Done with GW between {sample_1} and {sample_2} after"
                f" {round((end - start) / 60, 1)} minutes.",
                flush=True,
            )

        # Calculate entropic GW transport and distance
        if calculate_entr_gw:
            print(
                f"[STATUS] Calculating entr_GW between {sample_1} and {sample_2}.",
                flush=True,
            )
            start = time.time()
            entr_gw, entr_log = ot.gromov.entropic_gromov_wasserstein(
                d1,
                d2,
                p,
                q,
                "square_loss",
                epsilon=epsilon,
                log=True,
                verbose=False,
                max_iter=10000,
            )

            # Save results
            output_plan = output_dir.joinpath(f"entr_gw_plan_{sample_1}_{sample_2}.pkl")
            with open(output_plan, "wb") as f:
                pickle.dump(entr_gw, f)

            output_log = output_dir.joinpath(f"entr_gw_log_{sample_1}_{sample_2}.pkl")
            with open(output_log, "wb") as f:
                pickle.dump(entr_log, f)

            end = time.time()
            print(
                f"[STATUS] Done with entr_GW between {sample_1} and {sample_2} after"
                f" {round((end - start) / 60, 1)} minutes.",
                flush=True,
            )


def gw_embeddings(
    embeddings_dir,
    output_dir,
    distribution="balanced",
    alpha=0.5,
    calculate_gw=True,
    calculate_entr_gw=True,
    epsilon=5e-4,
    processes=8,
):
    """
    Calculate the (entropic) Gromov-Wasserstein transport plans and distances between
    all pairs of samples.

    Parameters
    ----------
    embeddings_dir : str or pathlib.Path object
        Path of the directory containing the embedding .csv files.
    output_dir : str or pathlib.Path object
        Path of the output directory where the results will be saved.
    distribution : 'balanced' or 'uniform'
        Whether the probability distribution of nodes is uniform, i.e. all nodes have
        the same probability of getting mapped, or the probability distribution of nodes
        is balanced between both node types, i.e., the sum of probabilities of all spot
        nodes is equal to the sum of probabilities of all cell type nodes. Default is
        `'balanced'`.
    alpha : float between 0 and 1
        Weight given to the cell type nodes in the probability distribution. If, e.g.,
        `alpha = 0.8`, the cell type nodes make up 80% of the total probability
        distribution and the spot nodes make up 20%. Only applies if
        `distribution = 'balanced'`. Default is `0.5`.
    calculate_gw : bool
        Whether to calculate the non-entropic GW transport plans and distances. Default
        is `True`.
    calculate_entr_gw : bool
        Whether to calculate the entropic GW transport plans and distances. Default is
        `True`.
    epsilon : float
        The epsilon value to be used for entropic GW. Default is `5e-4`.
    processes : int
        Number of worker processes to use. Default is `8`.
    """
    start = time.time()
    print(f"[STATUS] Beginning GW calculations.", flush=True)

    # Import all embeddings as dict of dataframes
    embeddings_path = Path(embeddings_dir)
    embeddings = {}
    for embedding in embeddings_path.iterdir():
        if ".csv" in embedding.name:
            sample_name = embedding.name.split("_embeddings")[0]
            embeddings[sample_name] = pd.read_csv(embedding, sep="\t", index_col=0)

    # Calculate probability distributions to use for OT plans
    distributions = {}
    for sample_name in embeddings.keys():
        if distribution == "uniform":
            num_nodes = len(embeddings[sample_name].index)
            node_probability = 1 / num_nodes
            dist = [node_probability for _ in range(num_nodes)]

        elif distribution == "balanced":
            index = list(embeddings[sample_name].index)
            # Count number of spots and celltypes in this sample
            num_spots = sum("spot" in node for node in index)
            num_celltypes = sum("celltype" in node for node in index)

            # Verify that sum of spots + sum of celltypes == sum of all nodes
            if not num_spots + num_celltypes == len(index):
                raise ValueError(
                    f"[ERROR] "
                    f"In sample {sample_name}: Encountered nodes with neither 'spot'"
                    f" nor 'celltype' (or both 'spot' and 'celltype') in name."
                    f" Counted {num_spots} spots and {num_celltypes} celltypes, but"
                    f" there are {len(index)} total nodes."
                )

            if not (0 <= alpha and alpha <= 1):
                raise ValueError(
                    f"[ERROR] alpha must be a float between 0 and 1 but is {alpha}."
                )
            # Define probability distribution so that cell type nodes make up fraction
            # alpha of the total distribution and spot nodes make up fraction 1-alpha
            ct_probability = alpha / num_celltypes
            spot_probability = (1 - alpha) / num_spots
            dist = [
                spot_probability if "spot" in node else ct_probability for node in index
            ]

        else:
            raise ValueError(
                f"[ERROR] distribution must be either 'balanced' or 'uniform' but is"
                f" '{distribution}'."
            )

        distributions[sample_name] = np.array(dist)

    # Calculate normalized distance kernels
    distance_matrices = {}
    for sample_name in embeddings.keys():
        distance_matrix = sp.spatial.distance.cdist(
            embeddings[sample_name], embeddings[sample_name]
        )
        distance_matrix /= distance_matrix.max()
        distance_matrices[sample_name] = distance_matrix

    # Prepare output directory
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    # Prepare array of all sample pairs
    sample_pairs = []
    for i, sample_1 in enumerate(distance_matrices.keys()):
        for j, sample_2 in enumerate(distance_matrices.keys()):
            if i <= j:
                sample_pairs.append((sample_1, sample_2))

    # Shuffle sample pairs
    random.shuffle(sample_pairs)

    # Prepare arguments for multiprocessing
    args = [
        (
            distance_matrices,
            distributions,
            sample_pairs[i::processes],
            output_dir,
            calculate_gw,
            calculate_entr_gw,
            epsilon,
        )
        for i in range(processes)
    ]

    # Calculate GW distances using multiprocessing
    with Pool(processes) as pool:
        pool.starmap(gw_pairs, args)

    end_prepro = time.time()
    seconds = round((end_prepro - start), 1)
    minutes = int(seconds // 60)
    rem_seconds = round(seconds % 60, 1)
    hours = int(minutes // 60)
    rem_minutes = round(minutes % 60, 1)
    print(
        f"[STATUS] Finished all GW calculations after {hours} hours {rem_minutes} mins"
        f" {rem_seconds} seconds.",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# RUNNING AS A SCRIPT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Gromov-Wasserstein transport plans and distances between all"
            " embedding pairs utilizing multiprocessing."
        )
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        help="Path of the directory containing the embedding .csv files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path of the output directory where the results will be saved.",
    )

    parser.add_argument(
        "--distribution",
        choices=["balanced", "uniform"],
        default="balanced",
        help=(
            "Whether the probability distribution of nodes is uniform, i.e. all nodes"
            " have the same probability of getting mapped, or the probability"
            " distribution of nodes is balanced between both node types, i.e., the sum"
            " of probabilities of all spot nodes is equal to the sum of probabilities"
            " of all cell type nodes."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            f"Weight given to the cell type nodes in the probability distribution. If,"
            f" e.g., alpha = 0.8, the cell type nodes make up 80% of the total"
            f" probability distribution and the spot nodes make up 20%. Only applies if"
            f" distribution = 'balanced'`. Default is `0.5`."
        ),
    )

    parser.add_argument(
        "--calculate_gw",
        action="store_true",
        help=(
            "Set this flag to calculate the non-entropic GW transport plans and"
            " distances."
        ),
    )

    parser.add_argument(
        "--calculate_entr_gw",
        action="store_true",
        help="Set this flag to calculate the entropic GW transport plans and distances.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0005,
        help="The epsilon value to be used for entropic GW. Default is 5e-4.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of worker processes to use. Default is 8.",
    )

    args = parser.parse_args()

    # Run GW
    gw_embeddings(
        args.embeddings_dir,
        args.output_dir,
        args.distribution,
        args.alpha,
        args.calculate_gw,
        args.calculate_entr_gw,
        epsilon=args.epsilon,
        processes=args.processes,
    )
