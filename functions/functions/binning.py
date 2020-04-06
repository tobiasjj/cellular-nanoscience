#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# binning, functions to bin data
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


def calculate_bin_means(data, bins=None, resolution=None, edges=None,
                        sortcolumn=0):
    """
    Calculate binned means.

    Parameters
    ----------
    data : 2D numpy.ndarray of type float
    bins : int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins. If bins
        is a sequence, it defines the bin edges, including the rightmost edge,
        allowing for non-uniform bin widths.

        If bins is a string from the list below, histogram_bin_edges will use
        the method chosen to calculate the optimal bin width and consequently
        the number of bins (see `numpy.histogram_bin_edges()` for more detail)
        from the data that falls within the requested range. While the
        bin width will be optimal for the actual data in the range, the number
        of bins will be computed to fill the entire range, including the empty
        portions. For visualisation, using the ‘auto’ option is suggested.
        Weighted data is not supported for automated bin size selection.

        ‘auto’
            Maximum of the ‘sturges’ and ‘fd’ estimators. Provides good all
            around performance.
        ‘fd’ (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into account
            data variability and data size.
        ‘doane’
            An improved version of Sturges’ estimator that works better with
            non-normal datasets.
        ‘scott’
            Less robust estimator that that takes into account data variability
            and data size.
        ‘stone’
            Estimator based on leave-one-out cross-validation estimate of the
            integrated squared error. Can be regarded as a generalization of
            Scott’s rule.
        ‘rice’
            Estimator does not take variability into account, only data size.
            Commonly overestimates number of bins required.
        ‘sturges’
            R’s default method, only accounts for data size. Only optimal for
            gaussian data and underestimates number of bins for large
            non-gaussian datasets.
        ‘sqrt’
            Square root (of data size) estimator, used by Excel and other
            programs for its speed and simplicity.

    Returns
    -------
    edges, centers, width, bin_means, bin_stds, bin_Ns
    """
    if bins is None:
        bins = number_of_bins(data[:, sortcolumn], resolution)
    if edges is None:
        # Create the bins based on data[:, sortcolumn]
        edges, centers, width, nbins = get_edges(data[:, sortcolumn], bins)
        # get first dim, i.e. the sortcolumn
        edges, centers, width = edges[0], centers[0], width[0]
    else:
        width = edges[1:] - edges[:-1]
        centers = width / 2 + edges[:-1]

    # Get the indices of the bins to which each value in input array belongs.
    bin_idx = np.digitize(data[:, sortcolumn], edges)

    # Find which points are on the rightmost edge.
    on_edge = data[:, sortcolumn] == edges[-1]
    # Shift these points one bin to the left.
    bin_idx[on_edge] -= 1

    # Calculate the histogram, means, and std of the data in the bins
    bin_Ns = np.array([np.sum(bin_idx == i)
                       for i in range(1, len(edges))])
    bin_means = np.array([data[bin_idx == i].mean(axis=0)
                          for i in range(1, len(edges))])
    bin_stds = np.array([data[bin_idx == i].std(axis=0, ddof=1)
                         for i in range(1, len(edges))])

    return edges, centers, width, bin_means, bin_stds, bin_Ns


def calculate_bin_means_ND(data, bins=None):
    """
    Calculate D-dimensional histogram

    Parameters
    ----------
    data : 2D np.ndarray of shape N,D
    bins : int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins. If bins
        is a sequence, it defines the bin edges, including the rightmost edge,
        allowing for non-uniform bin widths.

        If bins is a string from the list below, histogram_bin_edges will use
        the method chosen to calculate the optimal bin width and consequently
        the number of bins (see `numpy.histogram_bin_edges()` for more detail)
        from the data that falls within the requested range. While the
        bin width will be optimal for the actual data in the range, the number
        of bins will be computed to fill the entire range, including the empty
        portions. For visualisation, using the ‘auto’ option is suggested.
        Weighted data is not supported for automated bin size selection.

        ‘auto’
            Maximum of the ‘sturges’ and ‘fd’ estimators. Provides good all
            around performance.
        ‘fd’ (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into account
            data variability and data size.
        ‘doane’
            An improved version of Sturges’ estimator that works better with
            non-normal datasets.
        ‘scott’
            Less robust estimator that that takes into account data variability
            and data size.
        ‘stone’
            Estimator based on leave-one-out cross-validation estimate of the
            integrated squared error. Can be regarded as a generalization of
            Scott’s rule.
        ‘rice’
            Estimator does not take variability into account, only data size.
            Commonly overestimates number of bins required.
        ‘sturges’
            R’s default method, only accounts for data size. Only optimal for
            gaussian data and underestimates number of bins for large
            non-gaussian datasets.
        ‘sqrt’
            Square root (of data size) estimator, used by Excel and other
            programs for its speed and simplicity.

    Returns
    -------
    edges, centers, widths, bin_means, bin_stds, bin_Ns
    """
    N, D = data.shape
    edges, centers, widths, nbins = get_edges(data, bins)
    nbin = nbins + 2  # include outliers on each end

    # indices of x in bins of x and
    # indices of y in bins of y
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], data[:, i], side='right')
        for i in range(D)
    )
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (data[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.bincount(xy, minlength=nbin.prod())
    #bin_Ns = np.array([np.sum([xy == i])
    #                   for i in range(nbin.prod())])
    #data_xy = [data[xy == i]
    #           for i in range(nbin.prod())]
    mean_xy = np.array([np.mean(data[xy == i], axis=0)
                        for i in range(nbin.prod())])
    std_xy = np.array([np.std(data[xy == i], axis=0, ddof=1)
                       for i in range(nbin.prod())])

    # Shape into a proper matrix
    bin_Ns = hist.reshape(nbin)
    bin_Ns = bin_Ns.reshape(nbin)
    bin_means = mean_xy.T.reshape(np.r_[D, nbin]).T
    bin_stds = std_xy.T.reshape(np.r_[D, nbin]).T

    # remove outliers
    #hist = hist[1:-1, 1:-1]
    #mean_xy = mean_xy[1:-1, 1:-1]
    #std_xy = std_xy[1:-1, 1:-1]

    return edges, centers, widths, bin_means, bin_stds, bin_Ns


def get_edges(data, bins=None):
    """
    Get edges for bins in data

    Parameters
    ----------
    bins : int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins in the
        given range ('auto', by default). If bins is a sequence, it defines the
        bin edges, including the rightmost edge, allowing for non-uniform bin
        widths.

        If bins is a string from the list below, histogram_bin_edges will use
        the method chosen to calculate the optimal bin width and consequently
        the number of bins (see `numpy.histogram_bin_edges()` for more detail)
        from the data that falls within the requested range. While the
        bin width will be optimal for the actual data in the range, the number
        of bins will be computed to fill the entire range, including the empty
        portions. For visualisation, using the ‘auto’ option is suggested.
        Weighted data is not supported for automated bin size selection.

        ‘auto’
            Maximum of the ‘sturges’ and ‘fd’ estimators. Provides good all
            around performance.
        ‘fd’ (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into account
            data variability and data size.
        ‘doane’
            An improved version of Sturges’ estimator that works better with
            non-normal datasets.
        ‘scott’
            Less robust estimator that that takes into account data variability
            and data size.
        ‘stone’
            Estimator based on leave-one-out cross-validation estimate of the
            integrated squared error. Can be regarded as a generalization of
            Scott’s rule.
        ‘rice’
            Estimator does not take variability into account, only data size.
            Commonly overestimates number of bins required.
        ‘sturges’
            R’s default method, only accounts for data size. Only optimal for
            gaussian data and underestimates number of bins for large
            non-gaussian datasets.
        ‘sqrt’
            Square root (of data size) estimator, used by Excel and other
            programs for its speed and simplicity.

    Returns
    -------
    edges, centers, widths, nbins
    """
    if bins is None:
        bins = 'auto'

    if np.ndim(data) == 1:
        data = np.atleast_2d(data).T
    N, D = data.shape

    # bins is a str
    if isinstance(bins, str):
        bins = D*[bins]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an int
        bins = D*[bins]

    # Create edge arrays
    nbins = np.empty(D, int)
    edges = D*[None]
    centers = D*[None]
    widths = D*[None]

    for i in range(D):
        if np.ndim(bins[i]) == 0:
            # bins[i] is an int or a str
            edges[i] = np.histogram_bin_edges(data[:, i], bins[i])
        elif np.ndim(bins[i]) == 1:
            # bins[i] is a sequence
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                   '`bins[{}]` must be monotonically increasing, when an array'
                   .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))
        nbins[i] = len(edges[i]) - 1
        widths[i] = edges[i][1] - edges[i][0]
        centers[i] = edges[i][0:-1] + widths[i] / 2

    return edges, centers, widths, nbins


def number_of_bins(data, resolution=None):
    """
    Calculate the number of bins for requested resolution of the data

    If resolution is <= 0 this function returns None.

    Parameters
    ----------
    data : numpy.ndarray of type float
    resolution : float

    Returns
    -------
    bins : None or int
        Number of bins
    """
    if resolution is None:
        return None
    d_max = data.max()
    d_min = data.min()
    bins = int(round((d_max - d_min) * resolution))
    return max(1, bins)
