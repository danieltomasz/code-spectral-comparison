# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# %%
"""
Basic data analysis functions: pandas manipulation

"""


def lobes_long(raw, result):
    """transform channel info  from wide to long format"""

    psd_df = get_psd(raw)
    psd_df["id"] = psd_df.index
    psd_long = pd.melt(psd_df, id_vars=["id"])
    psd_long.columns = ["id", "frequency", "spectral_density"]
    # psd_long[['frequency','spectral_density']] = psd_long[['frequency','spectral_density']].astype(float)
    result_long = psd_long.set_index("id").join(result.set_index("ch_name"))
    return result_long


# %%
