import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pesco.experimental.plotting import plot_clusters  # noqa: E402


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
