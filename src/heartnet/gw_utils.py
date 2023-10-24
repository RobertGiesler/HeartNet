import math
import pickle
from pathlib import Path
import numpy as np


def gw_dist_from_pkl(gw_directory, symmetric=False, exclude=[]):
    """
    Read the (non-entropic) Gromov-Wasserstein distances of sample pairs from the output
    directory of a GW calculation.

    Parameters
    ----------
    gw_directory : str or pathlib.Path object
        Path of the GW output directory. Contains a subdirectory for the GW results of
        each sample pair, e.g., "GW_RZ_P3_FZ_P14".
    symmetric : bool
        Whether to return a symmetric dict, i.e., for every entry
        {("sample_1","sample_2") : gw_dist} there is also an entry
        {("sample_2","sample_1") : gw_dist} created. Default is False.
    exclude : list of str
        List of samples to exclude, e.g., ["RZ_P11", "control_P1"]. Default is the empty
        list.

    Returns
    -------
    dict
        Dictionary of the form {("sample_1","sample_2") : gw_dist}.
    """
    gw_dir = Path(gw_directory)
    gw_distances = {}

    if not gw_dir.is_dir():
        raise NotADirectoryError(
            f"[ERROR] gw_directory {gw_directory} is not a directory."
        )

    # Iterate over results of all sample pairs
    for sample_pair in gw_dir.iterdir():
        if sample_pair.is_dir():
            # Get names of samples ("sample_1", "sample_2")
            sample_names = split_gw_output_sample_names(sample_pair.name)

            if not (sample_names[0] in exclude or sample_names[1] in exclude):
                for pkl_file in sample_pair.iterdir():
                    if pkl_file.name.startswith("gw_log") and pkl_file.suffix == ".pkl":
                        with open(pkl_file, "rb") as f:
                            log = pickle.load(f)
                            gw_distances[(sample_names[0], sample_names[1])] = log[
                                "gw_dist"
                            ]
                            if symmetric:
                                gw_distances[(sample_names[1], sample_names[0])] = log[
                                    "gw_dist"
                                ]

    return gw_distances


def entr_gw_dist_from_pkl(gw_directory, symmetric=False, exclude=[]):
    """
    Read the entropic Gromov-Wasserstein distances of sample pairs from the output
    directory of a GW calculation.

    Parameters
    ----------
    gw_directory : str or pathlib.Path object
        Path of the GW output directory. Contains a subdirectory for the GW results of
        each sample pair, e.g., "GW_RZ_P3_FZ_P14".
    symmetric : bool
        Whether to return a symmetric dict, i.e., for every entry
        {("sample_1","sample_2") : entr_gw_dist} there is also an entry
        {("sample_2","sample_1") : entr_gw_dist} created. Default is False.
    exclude : list of str
        List of samples to exclude, e.g., ["RZ_P11", "control_P1"]. Default is the empty
        list.

    Returns
    -------
    dict
        Dictionary of the form {("sample_1","sample_2") : entr_gw_dist}.
    """
    gw_dir = Path(gw_directory)
    entr_gw_distances = {}

    if not gw_dir.is_dir():
        raise NotADirectoryError(
            f"[ERROR] gw_directory {gw_directory} is not a directory."
        )

    # Iterate over results of all sample pairs
    for sample_pair in gw_dir.iterdir():
        if sample_pair.is_dir():
            # Get names of samples ("sample_1", "sample_2")
            sample_names = split_gw_output_sample_names(sample_pair.name)

            if not (sample_names[0] in exclude or sample_names[1] in exclude):
                for pkl_file in sample_pair.iterdir():
                    if (
                        pkl_file.name.startswith("entr_gw_log")
                        and pkl_file.suffix == ".pkl"
                    ):
                        with open(pkl_file, "rb") as f:
                            entr_log = pickle.load(f)
                            entr_gw_distances[
                                (sample_names[0], sample_names[1])
                            ] = entr_log["gw_dist"]
                            if symmetric:
                                entr_gw_distances[
                                    (sample_names[1], sample_names[0])
                                ] = entr_log["gw_dist"]

    return entr_gw_distances


def split_gw_output_sample_names(gw_output_name):
    """
    Extract the sample names from the output directory name of a GW calculation for a
    sample pair.

    Parameters
    ----------
    gw_output_name : str
        Name of the GW output directory, e.g., "GW_control_P17_RZ_GT_P2".

    Returns
    -------
    2-tuple of str
        Names of the samples, e.g., ("control_P17", "RZ_GT_P2").
    """
    gw_output_name = Path(gw_output_name)
    name = gw_output_name.name.split("GW_")[-1]
    split = name.split("_")

    # Sample names
    sample_1 = ""
    sample_2 = ""

    # Variable to keep track of when to change to sample_2
    sample_switch = False

    # Iteratively add pieces of name to sample names
    for piece in split:
        if not sample_switch:
            sample_1 = sample_1 + piece + "_"
            if piece.startswith("P"):
                sample_switch = True

        else:
            sample_2 = sample_2 + piece + "_"

    # Remove extra "_"
    sample_1 = sample_1[:-1]
    sample_2 = sample_2[:-1]

    return (sample_1, sample_2)


def tuple_dict_to_2d_array(
    tuple_dict, sorted=True, symmetric=True, null_diagonal=False
):
    """
    Turn a dictionary with 2-tuples as keys into a 2-dimensional np.array.

    Parameters
    ----------
    tuple_dict : dict
        Dictionary with 2-tuples as keys.
    sorted : bool
        Whether to sort the index. Default is True.
    symmetric : bool
        Whether the data is symmetric. If True, then the returned array will have
        `array[x][y] == array[y][x]` for all indices `x` and `y`. Default is True.
    null_diagonal : bool
        Whether to set all values on the diagonal (i.e., distances between samples and
        themselves) to 0. Default is False.

    Returns
    -------
    array : np.array
        2-dimensional array with the dict values as entries. Missing entries will be set
        to zero.
    index : list
        Index of the array. E.g., if `index[0] == "key1"`, `index[1] == "key2"`, and
        `array[0][1] == 0.3`, then the input dict had an entry
        `{("key1","key2") : 0.3}`.
    """
    index = []
    for tuple_key in tuple_dict.keys():
        if not (type(tuple_key) == tuple and len(tuple_key) == 2):
            raise TypeError(
                f"[ERROR] All keys of tuple_dict must be 2-tuples but {tuple_key} is"
                f" not."
            )
        for key in tuple_key:
            if not key in index:
                index.append(key)

    if sorted:
        index.sort()

    res = np.zeros((len(index), len(index)))

    for i, key1 in enumerate(index):
        for j, key2 in enumerate(index):
            # If symmetric, we only have to fill half the values
            if symmetric and i > j:
                continue

            if i == j and null_diagonal:
                res[i][j] = 0
                continue

            try:
                value = tuple_dict[(key1, key2)]
            # No entry for (key1, key2)
            except KeyError:
                if symmetric:
                    # Check for entry (key2, key1)
                    try:
                        value = tuple_dict[(key2, key1)]
                    # No entry for either. Leave at zero and continue.
                    except KeyError:
                        continue
                # No entry for (key1, key2). Leave at zero and continue.
                else:
                    continue
            # If symmetric, check that mirrored value is equal or inexistent
            if symmetric:
                try:
                    mirrored_value = tuple_dict[(key2, key1)]
                except KeyError:
                    mirrored_value = value
                # (key1, key2) != (key2, key1) even though array should be symmetrical
                if mirrored_value != value:
                    # Catch NaN case
                    if math.isnan(mirrored_value) and math.isnan(value):
                        value = 0
                    else:
                        raise ValueError(
                            f"[ERROR] "
                            f"Entries [('{key1}', '{key2}')] and [('{key2}', '{key1}')] of"
                            f" tuple_dict do not match ({value} != {mirrored_value}) but"
                            f" the array is supposed to be symmetrical."
                        )

                # Values match
                res[i][j] = value
                res[j][i] = value
            else:
                res[i][j] = value

    return res, index
