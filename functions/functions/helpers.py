#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# helpers, functions to ease miscellaneous work
# Copyright 2016,2017,2018,2019 Tobias Jachowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import operator
import os
import sys

from contextlib import contextmanager

# Suppress stdout
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def min_max_idx(x, min_x=None, max_x=None, include_bounds=True,
                detailed=False):
    """
    Parameters
    ----------
    x : 1D numpy.ndarray of type float
        The x values.
    min_x : float
        The minimum value of `x`.
    max_x : float
        The maximum value of `x`.
    include_bounds : bool
        Whether to include or exlude min/max values in the output array

    Returns
    -------
    dict with indices
        The indices of the values to not be cropped (i.e. value is True).
    """
    # Select values with min < value < max:
    idx_min = compare_idx(x, min_x, 'greater', include_bounds=include_bounds)
    idx_max = compare_idx(x, max_x, 'less', include_bounds=include_bounds)
    idx_min_max = np.logical_and(idx_min, idx_max)

    if detailed:
        return_value = {
            'min_max': idx_min_max,
            'min': idx_min,
            'max': idx_max,
        }
    else:
        return_value = idx_min_max

    return return_value


_fu = {
    True: {
        'less': operator.le,
        'equal': operator.eq,
        'greater': operator.ge
    },
    False: {
        'less': operator.lt,
        'equal': operator.ne,
        'greater': operator.gt
}}
_va = {
    'less': float('inf'),
    'equal': 0.0,
    'greater': float('-inf')
}
def compare_idx(x, y=None, comparison='greater', include_bounds=True):
    f = _fu[include_bounds][comparison]
    y = _va[comparison] if y is None else y
    return f(x, y)


def step_idx(x, threshold, comparison='greater', include_bounds=True):
    idx = compare_idx(x, threshold, comparison, include_bounds=include_bounds)
    if np.any(idx):
        i_first = first_last_idx(idx)[0]
        idx[i_first:] = True
    return idx


def first_last_idx(idx_bool):
    # Get first and last index of values above min and values below max.
    length = len(idx_bool)
    i_first = np.argmax(idx_bool)
    i_last = length - 1 - np.argmax(idx_bool[::-1])
    return i_first, i_last


def make_contiguous_idx(idx_bool):
    # Make selection contiguous
    idx = idx_bool.copy()
    i_first, i_last = first_last_idx(idx)
    start = i_first
    stop = i_last + 1
    idx[start:stop] = True
    return idx


def crop_x_y_idx(x, y=None, min_x=None, max_x=None, min_y=None, max_y=None,
                 include_bounds=True):
    """
    Crop pairs of variates according to their minimum and maximum values.

    Parameters
    ----------
    x : 1D numpy.ndarray of type float
        The x values.
    y : 1D numpy.ndarray of type float
        The y values.
    min_x : float
        The minimum value of `x`.
    max_x : float
        The maximum value of `x`.
    min_y : float
        The minimum value of `y`.
    max_y : float
        The maximum value of `y`.
    include_bounds : bool
        Whether to include or exlude min/max values in the output arrays.

    Returns
    -------
    index array of type bool
        The index of the values to not be cropped (i.e. value is True).
    """
    idx_x = min_max_idx(
        x, min_x=min_x, max_x=max_x, include_bounds=include_bounds)
    idx_y = min_max_idx(
        y, min_x=min_y, max_x=max_y, include_bounds=include_bounds)

    idx = np.logical_and(idx_x, idx_y)
    return idx


def crop_x_y(x, y=None, min_x=None, max_x=None, min_y=None, max_y=None,
             include_bounds=True):
    """
    Crop pairs of variates according to their minimum and maximum values.

    Parameters
    ----------
    x : 1D numpy.ndarray of type float
        The x values.
    y : 1D numpy.ndarray of type float
        The y values.
    min_x : float
        The minimum value of `x`.
    max_x : float
        The maximum value of `x`.
    min_y : float
        The minimum value of `y`.
    max_y : float
        The maximum value of `y`.
    include_bounds : bool
        Whether to include or exlude min/max values in the output arrays.

    Returns
    -------
    tuple of 2 1D numpy.ndarray of type float
        The cropped values (x, y).
    """
    idx = crop_x_y_idx(x, y=y, min_x=min_x, max_x=max_x, min_y=min_y,
                       max_y=max_y, include_bounds=include_bounds)
    if y is None:
        return x[idx]
    else:
        return x[idx], y[idx]
