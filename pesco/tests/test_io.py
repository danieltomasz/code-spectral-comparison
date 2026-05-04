from pathlib import Path

import numpy as np

from pesco.io import (
    _load_mantini_centroids,
    _remove_frauscher_zero_buffers,
    load_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_SOURCES = PROJECT_ROOT / "data" / "Mantini2018"


def test_remove_frauscher_zero_buffers_keeps_sixty_seconds_per_patient():
    fs = 2.0
    data = np.array(
        [
            [
                1, 2, 0, 4, 0, 0, 0, 0, 5, 6,
                7, 8, 9, 10, 11, 12, 0, 0, 0, 0,
            ],
            [
                2, 3, 0, 5, 0, 0, 0, 0, 6, 7,
                8, 9, 10, 11, 12, 13, 0, 0, 0, 0,
            ],
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ],
        dtype=float,
    )
    patients = np.array([1, 1, 2])

    compact = _remove_frauscher_zero_buffers(
        data,
        patients,
        fs,
        duration=6.0,
        min_zero_run=2.0,
    )

    assert compact.shape == (3, 12)
    np.testing.assert_array_equal(
        compact[0],
        [1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    np.testing.assert_array_equal(
        compact[1],
        [2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    )
    np.testing.assert_array_equal(compact[2], np.arange(1, 13))


def test_load_mantini_centroids_has_region_hemisphere_coordinates():
    centroids = _load_mantini_centroids(DATA_PATH_SOURCES)

    assert len(centroids) == 76
    assert centroids[["region_number", "hemisphere"]].drop_duplicates().shape[0] == 76
    assert set(centroids["hemisphere"]) == {"L", "R"}
    assert not centroids[["mni_x", "mni_y", "mni_z"]].isna().any().any()


def test_load_sources_adds_mantini_centroid_coordinates():
    _, result = load_sources(DATA_PATH_SOURCES)

    expected_cols = [
        "region_number",
        "Region name",
        "Lobe",
        "hemisphere",
        "dataset",
        "mni_x",
        "mni_y",
        "mni_z",
    ]
    assert list(result.columns) == expected_cols
    assert not result[["mni_x", "mni_y", "mni_z"]].isna().any().any()

    right = result.loc["dataset01_01R", ["mni_x", "mni_y", "mni_z"]].to_numpy()
    left = result.loc["dataset01_01L", ["mni_x", "mni_y", "mni_z"]].to_numpy()
    assert result.loc["dataset01_01R", "hemisphere"] == "R"
    assert result.loc["dataset01_01L", "hemisphere"] == "L"
    assert not np.array_equal(right, left)


def test_load_sources_specific_skips_non_dataset_files():
    _, result = load_sources(DATA_PATH_SOURCES, specific=1)

    assert len(result) == 76
    assert result.index.str.startswith("dataset01_").all()
    assert not result[["mni_x", "mni_y", "mni_z"]].isna().any().any()
