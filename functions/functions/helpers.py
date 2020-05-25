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


def get_crop_idx(x, min_x=None, max_x=None, include_bounds=True, first_x=None,
                 last_x=None, switch_min_x=None, switch_max_x=None,
                 continuous_x=False):
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
    first_x: str
        The first index to be included. Possible values are: 'first_min',
        'last_min', 'first_max', or 'last_max'.
    last_x: str
        The last index to be included. Possible values are: 'first_min',
        'last_min', 'first_max', or 'last_max'.
    continuous_x: bool
        Include all indices between the first and the last possible index.

    Returns
    -------
    dict with indices
        The index of the values to not be cropped (i.e. value is True).
    """
    length_x = len(x)
    # Reduce calculation time, if no min/max values are given
    if min_x is None and max_x is None:
        idx = np.ones_like(x, dtype=bool)
        return_values = {
            'first_min': 0,
            'last_min': length_x - 1,
            'first_max': 0,
            'last_max': length_x - 1,
            'idx_min_max': idx,
            'idx_crop': idx
        }
    else:
        if include_bounds:
            og = operator.ge
            ol = operator.le
        else:
            og = operator.gt
            ol = operator.lt
        min_x = min_x or float('-inf')
        max_x = max_x or float('inf')
        switch_min_x = switch_min_x or float('inf')
        switch_max_x = switch_max_x or float('-inf')
        i_min_x = og(x, min_x)
        i_max_x = ol(x, max_x)
        idx_min_max = np.logical_and(i_min_x, i_max_x)
        i_on_min_x = og(x, switch_min_x)
        i_on_max_x = ol(x, switch_max_x)
        i_on_x = np.logical_or(i_on_min_x, i_on_max_x)

        # Calculate first and last index of values above min and values below
        # max.
        # Ignore values after first time above ignore_switch_min
        def get_first_last_idx(i_bool):
            length = len(i_bool)
            i_first = np.argmax(i_bool)
            i_last = length - 1 - np.argmax(i_bool[::-1])
            i_not_last = length - 1 - np.argmax(np.logical_not(i_bool[::-1]))
            return i_first, i_last, i_not_last

        if np.any(i_on_x):
            i_first_on = get_first_last_idx(i_on_x)[0]
        else:
            i_first_on = length_x - 1
        i_first_min, i_last_min, i_last_not_min \
            = get_first_last_idx(i_min_x[:i_first_on + 1])
        i_first_max, i_last_max, i_last_not_max \
            = get_first_last_idx(i_max_x[:i_first_on + 1])
        return_values = {
            'first_min': i_first_min,
            'first_max': i_first_max,
            'last_min': i_last_not_min + 1,
            'last_max': i_last_not_max + 1
        }

        # Determine start and stop index
        i_first_min_max, i_last_min_max = get_first_last_idx(idx_min_max)[:2]
        start_x = return_values.get(first_x, i_first_min_max)
        stop_x = return_values.get(last_x, i_last_min_max) + 1

        return_values['idx_min_max'] = idx_min_max

        idx = idx_min_max.copy()
        idx[:start_x] = False
        idx[stop_x:] = False
        if continuous_x:
            idx[start_x:stop_x] = True
        return_values['idx_crop'] = idx

    return return_values


def crop_x_y_idx(x, y=None, min_x=None, max_x=None, min_y=None, max_y=None,
                 include_bounds=True, first_x=None, last_x=None, first_y=None,
                 last_y=None, continuous_x=False, continuous_y=False):
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
    idx_x = get_crop_idx(
        x, min_x=min_x, max_x=max_x, include_bounds=include_bounds,
        first_x=first_x, last_x=last_x, continuous_x=continuous_x)
    idx_y = get_crop_idx(
        y, min_x=min_y, max_x=max_y, include_bounds=include_bounds,
        first_x=first_y, last_x=last_y, continuous_x=continuous_y)

    idx = np.logical_and(idx_x['idx_crop'], idx_y['idx_crop'])
    return idx


def crop_x_y(x, y=None, min_x=None, max_x=None, min_y=None, max_y=None,
             include_bounds=True, first_x=None, last_x=None, first_y=None,
             last_y=None, continuous_x=False, continuous_y=False):
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
                       max_y=max_y, include_bounds=include_bounds,
                       first_x=first_x, last_x=last_x, first_y=first_y,
                       last_y=last_y, continuous_x=continuous_x,
                       continuous_y=continuous_y)
    if y is None:
        return x[idx]
    else:
        return x[idx], y[idx]
