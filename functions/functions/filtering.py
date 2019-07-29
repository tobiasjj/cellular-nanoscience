#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# filtering, functions to filter data
# Copyright 2019 Tobias Jachowski
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
from scipy.ndimage import convolve1d
from scipy.ndimage import median_filter

from ...stepfinder.stepfinder import filter_fbnl


def filter_fbnl_data(data, resolution, filter_time=0.005, edginess=1):
    # Filter the data
    window = window_var = max(int(np.round(filter_time * resolution)), 1)
    cap_data = True

    if data.ndim == 1:
        data = data[np.newaxis].T
    fbnl_filters = [None]*data.shape[1]
    filtered_data = np.empty_like(data)
    for t in range(data.shape[1]):  # traces ...
        d = data[:, t]
        fbnl_filter = filter_fbnl(d, resolution, window=window,
                                  window_var=window_var, p=edginess,
                                  cap_data=cap_data)
        filtered_data[:, t] = fbnl_filter.data_filtered
        fbnl_filters[t] = fbnl_filter

    return filtered_data, fbnl_filters


def filter_fbnl_region(experiment, region, traces, time=False, tmin=None,
                       tmax=None, filter_time=0.005, edginess=1):
    # Get the data
    resolution = experiment.region(region).samplingrate
    if tmin is None:
        start = tmin
    else:
        start = int(round(tmin*resolution))
    if tmax is None:
        stop = None
    else:
        stop = int(round(tmax*resolution)) + 1
    samples = slice(start, stop)
    data = experiment.region(region).get_data(traces=traces, samples=samples,
                                              time=time)

    if time:
        d = data[:,1:]
    else:
        d = data

    # filter the data
    data_filtered, fbnl_filters = filter_fbnl_data(d, resolution, filter_time,
                                                   edginess)

    if time:
        data_filtered = np.c_[data[:,0], data_filtered]

    return data, data_filtered, fbnl_filters


def moving_filter(data, window, moving_filter='mean', mode='reflect', cval=0.0,
                  origin=0):
    """
    Apply a moving filter to data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be filterd.
    window : int
        The window size of the moving filter.
    moving_filter : str, optional
        The filter to be used for the moving filter. Can be one of 'mean' or
        'median'.
    mode : str, optional
        mode       |   Ext   |         Data           |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
        See 'scipy.ndimage.convolve1d' or 'scipy.ndimage.median_filter'
    cval : float, optional
        See 'scipy.ndimage.convolve1d' or 'scipy.ndimage.median_filter'
    origin : int, optional
        See 'scipy.ndimage.convolve1d' or 'scipy.ndimage.median_filter'
    """
    mode = mode or 'reflect'
    cval = cval or 0.0
    origin = origin or 0
    if moving_filter == 'mean' or moving_filter == 'average':
        return movingmean(data, window, mode=mode, cval=cval, origin=origin)
    else:  # if moving == 'median'
        return movingmedian(data, window, mode=mode, cval=cval, origin=origin)


def movingmean(data, window, mode='reflect', cval=0.0, origin=0):
    weights = np.repeat(1.0, window)/window
    # sma = np.zeros((data.shape[0] - window + 1, data.shape[1]))
    sma = convolve1d(data, weights, axis=0, mode=mode, cval=cval,
                     origin=origin)
    return sma


def movingmedian(data, window, mode='reflect', cval=0.0, origin=0):
    if data.ndim == 1:
        size = window
    else:
        size = (window, 1)
    smm = median_filter(data, size=size, mode=mode, cval=cval, origin=origin)
    return smm


def moving_mean(data, window):
    """
    Calculate a filtered signal by using a moving mean. The first datapoint is
    the mean of the first `window` datapoints and the last datapoint is the
    mean of the last `window` datapoints of the original data. This function
    does not handle the lost edges of the data, i.e. the filtered data is
    shortened by `window` datapoints.

    This function is faster than the function `movingmean()`.

    Parameters
    ----------
    data : 1D numpy.ndarray of type float
        Data to calculate the rolling mean from.
    window : int
        Length of the window to calculate the rolling mean with.

    Returns
    -------
    1D numpy.ndarray of type float
        The data filtered with a rolling mean.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window
