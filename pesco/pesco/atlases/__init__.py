"""Custom brain atlases for ggsegpy."""

from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

_DATA = files(__package__) / "data"


def _data_path(name: str) -> Path:
    return Path(str(_DATA / name))


def miccai_paths() -> tuple[Path, Path]:
    """Return (core_parquet, 2d_parquet) paths for the bundled MICCAI atlas."""
    return _data_path("miccai_core.parquet"), _data_path("miccai_2d.parquet")


def frauscher_paths() -> tuple[Path, Path]:
    """Return (core_parquet, 2d_parquet) paths for the Frauscher 38-region atlas."""
    return _data_path("frauscher_core.parquet"), _data_path("frauscher_2d.parquet")


def _build_atlas(name: str, core_p: Path, ggseg_p: Path):
    from ggsegpy.atlas import CorticalAtlas
    from ggsegpy.data import CorticalData

    core = pd.read_parquet(core_p)
    df = pd.read_parquet(ggseg_p)
    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    df = df.drop(columns=["geometry_wkt"]).merge(core, on="label", how="left")
    ggseg2d = gpd.GeoDataFrame(df, geometry="geometry")

    ggseg3d_empty = pd.DataFrame(
        {
            "label": pd.Series(dtype=str),
            "vertex_indices": pd.Series(dtype=object),
        }
    )

    return CorticalAtlas(
        atlas=name,
        type="cortical",
        core=core,
        data=CorticalData(ggseg=ggseg2d, ggseg3d=ggseg3d_empty, mesh=None),
        palette=dict(zip(core["label"], core["color"])),
    )


def load_miccai():
    """Build a ggsegpy `CorticalAtlas` for the MICCAI (Neuromorphometrics) atlas."""
    return _build_atlas("miccai", *miccai_paths())


def load_frauscher():
    """Build a ggsegpy `CorticalAtlas` for the Frauscher 38-region atlas
    (MICCAI labels aggregated per Frauscher 2018 grouping)."""
    return _build_atlas("frauscher38", *frauscher_paths())


__all__ = [
    "load_miccai",
    "load_frauscher",
    "miccai_paths",
    "frauscher_paths",
]
