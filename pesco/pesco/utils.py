#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:22:52 2019

@author: daniel
"""

import numpy as np


def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """

    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys > 0:
        d = {
            k: d[k] for k in list(d.keys())[:nkeys]
        }  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.items():  # iteritems loops through key, value pairs
            print("\t" * indent + "Key: " + str(key))
            print_mat_nested(value, indent + 1)

    if (
        isinstance(d, np.ndarray) and d.dtype.names is not None
    ):  # Note: and short-circuits by default
        for n in d.dtype.names:  # This means it's a struct, it's bit of a kludge test.
            print("\t" * indent + "Field: " + str(n))
            print_mat_nested(d[n], indent + 1)

