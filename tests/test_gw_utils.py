from heartnet import gw_utils

import numpy as np
import pytest


def test_split_gw_output_sample_names():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST NAMES AND TARGET SPLITS
    # ----------------------------------------------------------------------------------
    test_names = [
        "GW_FZ_GT_P19_RZ_BZ_P3",
        "GW_RZ_P9_IZ_P10",
        "GW_RZ_P3_GT_IZ_P15",
        "GW_control_P1_control_P8",
        "GW_control_P1_GT_IZ_P9",
        "GW_RZ_BZ_P12_GT_IZ_P9",
    ]

    target_splits = [
        ("FZ_GT_P19", "RZ_BZ_P3"),
        ("RZ_P9", "IZ_P10"),
        ("RZ_P3", "GT_IZ_P15"),
        ("control_P1", "control_P8"),
        ("control_P1", "GT_IZ_P9"),
        ("RZ_BZ_P12", "GT_IZ_P9"),
    ]

    # ----------------------------------------------------------------------------------
    # COMPUTE SPLITS
    # ----------------------------------------------------------------------------------
    results = []
    for name in test_names:
        results.append(gw_utils.split_gw_output_sample_names(name))

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    for i, res in enumerate(results):
        print(f"input name: {test_names[i]}")
        print(f"target: {target_splits[i]}")
        print(f"result: {res}")
        assert res == target_splits[i]


def test_tuple_dict_to_2d_array_unsorted_asymmetric():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST DICT AND TARGET ARRAY AND INDEX
    # ----------------------------------------------------------------------------------
    tuple_dict = {
        ("a", "c"): 1,
        ("a", "b"): 2.2,
        ("b", "c"): 0.4,
        ("b", "a"): 0.33,
        ("b", "b"): 5,
    }

    target_index = ["a", "c", "b"]

    target_array = np.array(
        [
            [0, 1.0, 2.2],
            [0, 0, 0],
            [0.33, 0.4, 5.0],
        ]
    )

    # ----------------------------------------------------------------------------------
    # COMPUTE 2D-ARRAY AND INDEX
    # ----------------------------------------------------------------------------------
    res_array, res_index = gw_utils.tuple_dict_to_2d_array(
        tuple_dict, sorted=False, symmetric=False
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target index: {target_index}")
    print(f"result index: {res_index}")
    assert res_index == target_index

    print(f"target array: {target_array}")
    print(f"result array: {res_array}")
    assert np.array_equal(res_array, target_array)


def test_tuple_dict_to_2d_array_sorted_symmetric():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST DICT AND TARGET ARRAY AND INDEX
    # ----------------------------------------------------------------------------------
    tuple_dict = {
        ("a", "c"): 1,
        ("a", "b"): 2.2,
        ("b", "c"): 0.4,
        ("c", "b"): 0.4,
        ("b", "b"): 5,
    }

    target_index = ["a", "b", "c"]

    target_array = np.array(
        [
            [0, 2.2, 1.0],
            [2.2, 5.0, 0.4],
            [1.0, 0.4, 0],
        ]
    )

    # ----------------------------------------------------------------------------------
    # COMPUTE 2D-ARRAY AND INDEX
    # ----------------------------------------------------------------------------------
    res_array, res_index = gw_utils.tuple_dict_to_2d_array(
        tuple_dict, sorted=True, symmetric=True
    )

    # ----------------------------------------------------------------------------------
    # ASSERT RESULTS
    # ----------------------------------------------------------------------------------
    print(f"target index: {target_index}")
    print(f"result index: {res_index}")
    assert res_index == target_index

    print(f"target array: {target_array}")
    print(f"result array: {res_array}")
    assert np.array_equal(res_array, target_array)


def test_tuple_dict_to_2d_array_symmetric_error():
    # ----------------------------------------------------------------------------------
    # PREPARE TEST DICT AND TARGET ARRAY AND INDEX
    # ----------------------------------------------------------------------------------
    tuple_dict = {
        ("a", "c"): 1,
        ("a", "b"): 2.2,
        ("b", "c"): 0.4,
        ("c", "b"): 0.33,  # Should raise error because ('b','c') != ('c','b')
        ("b", "b"): 5,
    }

    # ----------------------------------------------------------------------------------
    # ASSERT ERROR
    # ----------------------------------------------------------------------------------
    with pytest.raises(ValueError):
        res_array, res_index = gw_utils.tuple_dict_to_2d_array(
            tuple_dict, sorted=True, symmetric=True
        )
