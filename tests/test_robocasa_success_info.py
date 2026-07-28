import numpy as np

from examples.train_robomimic import _extract_success_info


def _object_success(*items):
    payload = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        payload[index] = np.asarray(item, dtype=bool)
    return payload


def test_extract_success_info_reduces_nested_object_array():
    info = {"success": _object_success([False, False])}
    np.testing.assert_array_equal(
        _extract_success_info(info, 1), np.asarray([False])
    )

    info = {"success": _object_success([False, True])}
    np.testing.assert_array_equal(
        _extract_success_info(info, 1), np.asarray([True])
    )


def test_extract_success_info_preserves_per_env_mapping():
    info = {
        "success": _object_success(
            [False, False],
            [False, True],
            [True, False],
        )
    }
    np.testing.assert_array_equal(
        _extract_success_info(info, 3),
        np.asarray([False, True, True]),
    )


def test_extract_success_info_handles_scalar_and_missing_values():
    np.testing.assert_array_equal(
        _extract_success_info({"success": True}, 2),
        np.asarray([True, True]),
    )
    np.testing.assert_array_equal(
        _extract_success_info({}, 2),
        np.asarray([False, False]),
    )


def test_extract_success_info_handles_nested_mappings_and_object_scalars():
    nested = {
        "success": np.asarray(
            {
                "task": _object_success([False, True]),
                "irrelevant": [False, False],
            },
            dtype=object,
        )
    }
    np.testing.assert_array_equal(
        _extract_success_info(nested, 1),
        np.asarray([True]),
    )


def test_extract_success_info_does_not_coerce_ragged_payload_as_array():
    info = {
        "success": [
            np.asarray([False]),
            np.asarray([False, True]),
        ]
    }
    np.testing.assert_array_equal(
        _extract_success_info(info, 1),
        np.asarray([True]),
    )
