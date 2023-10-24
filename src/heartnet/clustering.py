import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import squareform
from heartnet import gw_utils
from pathlib import Path
import pandas as pd


def agglomerative_hierarchical_clustering(
    gw_directory, gt_labels_dict, method="ward", n_clusters=2, exclude=[]
):
    """
    Cluster the samples using agglomerative hierarchical clustering based on the Gromov-
    Wasserstein distances between them. Clustering is done for both GW and entropic GW
    distances. Results are returned as Pandas DataFrames.

    Parameters
    ----------
    gw_directory : str or pathlib.Path object
        Path of the GW output directory. Contains a subdirectory for the GW results of
        each sample pair, e.g., "GW_RZ_P3_FZ_P14".
    gt_labels_dict : dict
        Ground truth label dictionary of the form {sample_name : true_label}.
    method : 'single', 'complete', 'average', or 'ward'
        The linkage algorithm to use. See scipy linkage methods for more information.
        Default is 'ward'.
    n_clusters : int
        Number of clusters to group the samples into. Default is 2.
    exclude : list of str
        List of samples to exclude, e.g., ["RZ_P11", "control_P1"]. Default is the empty
        list.

    Returns
    -------
    DataFrame : gw_clustering
        DataFrame with sample names as index and columns "labels_pred" for predicted
        cluster and "labels_true" for ground truth cluster. If gw_directory contains no
        data for GW distances, None is returned.
    DataFrame : entr_gw_clustering
        DataFrame with sample names as index and columns "labels_pred" for predicted
        cluster and "labels_true" for ground truth cluster. If gw_directory contains no
        data for entropic GW distances, None is returned.
    """
    gw_directory = Path(gw_directory)

    # Get GW distances
    gw_distances_dict = gw_utils.gw_dist_from_pkl(gw_directory, exclude=exclude)
    entr_gw_distances_dict = gw_utils.entr_gw_dist_from_pkl(
        gw_directory, exclude=exclude
    )

    # Turn into np.array
    gw_distances_array, index = gw_utils.tuple_dict_to_2d_array(
        gw_distances_dict, symmetric=True
    )
    entr_gw_distances_array, entr_index = gw_utils.tuple_dict_to_2d_array(
        entr_gw_distances_dict, symmetric=True
    )

    # Initialize return values as None
    gw_clustering = None
    entr_gw_clustering = None

    # Cluster samples based on GW distances
    if gw_distances_array.size > 0:
        # Convert distance matrix into condensed form
        condensed_dist_mat = squareform(gw_distances_array, checks=False)

        # Ground truth labels
        true_labels = []
        for sample in index:
            true_labels.append(gt_labels_dict[sample])

        # Perform hierarchical clustering
        linkage = hierarchy.linkage(
            condensed_dist_mat, optimal_ordering=False, method=method
        )
        predicted_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters)

        # Prepare results DataFrame
        gw_clustering = pd.DataFrame(
            data=predicted_labels, columns=["labels_pred"], index=index
        )
        gw_clustering["labels_true"] = true_labels

    # Cluster samples based on Entropic GW distances
    if entr_gw_distances_array.size > 0:
        # Convert distance matrix into condensed form
        condensed_dist_mat = squareform(entr_gw_distances_array, checks=False)

        # Ground truth labels
        true_labels = []
        for sample in entr_index:
            true_labels.append(gt_labels_dict[sample])

        # Perform hierarchical clustering
        linkage = hierarchy.linkage(
            condensed_dist_mat, optimal_ordering=False, method=method
        )
        predicted_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters)

        # Prepare results DataFrame
        entr_gw_clustering = pd.DataFrame(
            data=predicted_labels, columns=["labels_pred"], index=entr_index
        )
        entr_gw_clustering["labels_true"] = true_labels

    # Return results DataFrames
    return gw_clustering, entr_gw_clustering
