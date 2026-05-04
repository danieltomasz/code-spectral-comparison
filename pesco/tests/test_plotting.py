import sys
import types

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pesco.experimental.plotting import plot_cluster_brain, plot_clusters  # noqa: E402


def test_plot_clusters_can_use_log_y_axis(tmp_path):
    f = np.array([0.5, 1.0, 2.0, 4.0])
    psd_clust = pd.DataFrame(
        {
            0.5: [0.1, 0.2, 0.05, 0.08],
            1.0: [0.08, 0.1, 0.04, 0.05],
            2.0: [0.04, 0.05, 0.02, 0.03],
            4.0: [0.02, 0.03, 0.01, 0.02],
            "clusters": [0, 0, 1, 1],
        }
    )

    fig, ax = plot_clusters(
        psd_clust,
        f,
        "test",
        output_dir=tmp_path,
        log_y=True,
    )

    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_plot_cluster_brain_can_use_label_markers_with_viridis(monkeypatch):
    calls = {}

    class FakeDisplay:
        def __init__(self):
            self.added = []

        def add_markers(self, **kwargs):
            self.added.append(kwargs)

    display = FakeDisplay()

    def fake_plot_markers(**kwargs):
        calls["plot_markers"] = kwargs
        return display

    fake_nilearn = types.ModuleType("nilearn")
    fake_plotting = types.ModuleType("nilearn.plotting")
    fake_plotting.plot_markers = fake_plot_markers
    fake_nilearn.plotting = fake_plotting
    monkeypatch.setitem(sys.modules, "nilearn", fake_nilearn)
    monkeypatch.setitem(sys.modules, "nilearn.plotting", fake_plotting)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    labels = np.array([0, 1, 0])

    plot_cluster_brain(
        positions,
        labels,
        label_order=[0, 1],
        label_markers={0: "^"},
        node_size=12,
        show=False,
    )

    viridis = plt.get_cmap("viridis")
    base_call = calls["plot_markers"]
    assert base_call["node_kwargs"]["marker"] == "o"
    np.testing.assert_array_equal(base_call["node_values"], np.array([1.0]))

    triangle_layer = display.added[0]
    assert triangle_layer["marker"] == "^"
    assert triangle_layer["marker_color"] == [viridis(0), viridis(0)]

    legend = plt.gcf().legends[0]
    assert legend.legend_handles[0].get_marker() == "^"
    plt.close("all")
