#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# force_extension, functions to work with and show force extension curves
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

import collections
import itertools
import math
import numpy as np
import os
import unzipping_simulation as uzsi

from IPython.display import display
from ipywidgets import Label, interactive_output, Checkbox, IntText, BoundedIntText, FloatText, HBox, VBox
from matplotlib import pyplot as plt
from stepfinder import filter_fbnl

from .binning import calculate_bin_means

def cart2sph(x, y, z, offset_phi=0, positive_phi=False):
    """
    offset_phi : float
        angle in Euclidian plane that should point in the direction of positive x
    """
    # cart2sph -- Transform Cartesian to spherical coordinates
    # Spherical coordinates (r, θ, φ) as commonly used in physics (ISO convention):
    # radial distance r, inclination θ (theta), and azimuth φ (phi).
    hxy = math.hypot(x, y)
    r = math.hypot(hxy, z)
    theta = math.atan2(hxy, z)
    phi = math.atan2(y, x) - offset_phi
    if positive_phi and phi < 0:
        phi += 2 * math.pi
    return r, theta, phi


def sph2cart(r, theta, phi, offset_phi=0):
    """
    offset_phi : float
        angle in Euclidian plane that points in the directon of positive x
    """
    # sph2cart -- Transform spherical to Cartesian coordinates
    # Spherical coordinates (r, θ, φ) as commonly used in physics (ISO convention):
    # radial distance r, inclination θ (theta), and azimuth φ (phi).
    phi += offset_phi
    rsin_theta = r * math.sin(theta)
    x = rsin_theta * math.cos(phi)
    y = rsin_theta * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def angle(v1, v2):
    # angle between two vectors
    #return math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))
    # does not work as well for small angles, but is faster:
    cos_theta = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = max(-1, cos_theta)
    cos_theta = min(1, cos_theta)
    return math.acos(cos_theta)


def _get_speed_approx(tether, i, cycle=None):
    cycle = 'stress' if cycle is None else cycle
    pair = tether.stress_release_pairs(i=i, info=True)
    j = 0 if cycle == 'stress' else 1
    idx = pair[j][0]
    ax = pair[2 + j][0,0]
    trace = {'x': 'positionX', 'y': 'positionY'}
    position = tether.get_data(traces=trace[ax], samples=idx)
    amplitude = position.max() - position.min()
    duration = (idx.stop - idx.start) / tether.resolution
    speed = amplitude / duration  # m / s
    return speed


def binned_force_extension(tether, i, posmin=10e-9, bins=None, resolution=None,
                           bin_width_e=None, sortcolumn=0, fXYZ_factors=None,
                           angles=False, extra_traces=None,
                           angles_after_binning=False, phi_shift_twopi=False):
    """
    Parameters
    ----------
    bins : int or str
        number of bins, takes precedence over resolution
    resolution : float
        number of bins per unit of sortcolumn.
    bin_width_e : float
        Width of bins of extension in m. Only evaluated if bins and resolution
        are None. A resolution (s) is calculated by dividing the bin_width_e
        with an approximate speed of the positionXY signal. Therefore, if
        bin_width_e is evaluated, the sortcolumn is automatically set to 0
        (i.e. time).
    sortcolumn : int
        0: time, 1: extension, 2: force, n >= 3: angles and/or extra_columns
    angles : bool
        Calculate theta and phi for extension and force.
        3: theta_extension, 4: phi_extension, 5: theta_force, 6: phi_force,
        7,8,9: distanceXYZ, 10,11,12: forceXYZ
    angles_after_binning : bool
        13,14,15,16: theta (13,15) and phi (14,16) for extension and force
    """
    fe_pair = tether.force_extension_pair(i=i, time=True, posmin=posmin,
                                          fXYZ_factors=fXYZ_factors)
    (x_stress, y_stress, info_stress,
     x_release, y_release, info_release,
     t_stress, t_release) = fe_pair

    data = [None]*2  # 0: stress, 1: release
    data[0] = np.c_[t_stress, x_stress, y_stress]
    data[1] = np.c_[t_release, x_release, y_release]

    if angles or extra_traces is not None:
        # Get stress/release indices of stress/release pair i
        idxs = tether.stress_release_pairs(i=i)

        # Calculate angles for stress and release of extension and force
        # vectors
        for c, idx in enumerate(idxs):  # 1: stress, 2: release
            if  angles:
                # Get distance/force (vectors XYZ)
                distanceXYZ = tether.distanceXYZ(samples=idx[0])
                forceXYZ = tether.forceXYZ(samples=idx[0],
                                           fXYZ_factors=fXYZ_factors)
                # calculate angles theta and phi
                angle_extension = np.array([
                    cart2sph(*point)[1:] for point in distanceXYZ
                    ])*180/math.pi
                angle_force = np.array([
                    cart2sph(*point)[1:] for point in forceXYZ
                    ])*180/math.pi
                if phi_shift_twopi:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                data[c] = np.c_[data[c], angle_extension, angle_force,
                                distanceXYZ, forceXYZ]
            if extra_traces is not None:
                extra_data = tether.get_data(traces=extra_traces,
                                             samples=idx[0])
                data[c] = np.c_[data[c], extra_data]

    edges = [None]*2
    centers = [None]*2
    widths = [None]*2
    bin_means = [None]*2
    bin_stds = [None]*2
    bin_Ns = [None]*2
    for c, cycle in enumerate(['stress', 'release']):  # 0=stress, 1=release
        if bins is None and resolution is None and bin_width_e is not None:
            speed = _get_speed_approx(tether, i, cycle)
            resolution = speed / bin_width_e
            sortcolumn = 0
        result = calculate_bin_means(data[c], bins=bins, resolution=resolution,
                                     sortcolumn=sortcolumn)
        (edges[c], centers[c], widths[c],
         bin_means[c], bin_stds[c], bin_Ns[c]) = result

    # Calculate angles with already binned distance/force data
    if angles and angles_after_binning:
        for c in range(2):  # 1: stress, 2: release
            angle_extension = np.array([
                cart2sph(*point)[1:]
                for point in bin_means[c][:, 7:10]])*180/math.pi
            angle_force = np.array([
                cart2sph(*point)[1:]
                for point in bin_means[c][:, 10:13]])*180/math.pi
            if phi_shift_twopi:
                angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                angle_force[angle_force[:,1] < 0.0, 1] += 360
            bin_means[c] = np.c_[bin_means[c], angle_extension, angle_force]

    return edges, centers, widths, bin_means, bin_stds, bin_Ns


def fbnl_force_extension(tether, i, posmin=10e-9, filter_time=None,
                         filter_length_e=None, edginess=1, fXYZ_factors=None,
                         angles=False, extra_traces=None,
                         angles_after_filter=False, phi_shift_twopi=False):
    """
    Parameters
    ----------
    filter_time : float
        time of running filter in s
    filter_length_e : float
        Length of running filter of extension in m. Only evaluated if
        filter_time is None. A filter_time (s) is calculated by dividing the
        filter_length_e with an approximate speed of the positionXY signal.
    angles : bool
        Calculate theta and phi for extension and force.
        3: theta_extension, 4: phi_extension, 5: theta_force, 6: phi_force,
        7,8,9: distanceXYZ, 10,11,12: forceXYZ
    angles_after_filter : bool
        13,14,15,16: theta (13,15) and phi (14,16) for extension and force

    Returns
    -------
    filtered_data, fbnl_filters
    filtered_data is a list of two np.ndarrays (0: stress, 1: release)
        each array has the filtered data with the columns 0: time, 1: extension
        2: force, and extra traces/angles
        fbnl_filters is a list of two lists (0: stress, 1: release) containing
        the individual FBNL_Filter_results of the filtered data
    """
    fe_pair = tether.force_extension_pair(i=i, time=True, posmin=posmin,
                                          fXYZ_factors=fXYZ_factors)
    (x_stress, y_stress, info_stress,
     x_release, y_release, info_release,
     t_stress, t_release) = fe_pair

    data = [None]*2
    data[0] = np.c_[t_stress, x_stress, y_stress]
    data[1] = np.c_[t_release, x_release, y_release]
    if angles or extra_traces is not None:
        # Get stress/release indices of stress/release pair i
        idxs = tether.stress_release_pairs(i=i)

        # Calculate angles for stress and release of extension and force
        # vectors
        for c, idx in enumerate(idxs):  # 1: stress, 2: release
            if angles:
                # Get distance/force (vectors XYZ)
                distanceXYZ = tether.distanceXYZ(samples=idx[0])
                forceXYZ = tether.forceXYZ(samples=idx[0],
                                           fXYZ_factors=fXYZ_factors)
                # calculate angles theta and phi
                angle_extension = np.array([
                    cart2sph(*point)[1:] for point in distanceXYZ
                    ])*180/math.pi
                angle_force = np.array([
                    cart2sph(*point)[1:] for point in forceXYZ
                    ])*180/math.pi
                if phi_shift_twopi:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                data[c] = np.c_[data[c], angle_extension, angle_force,
                                distanceXYZ, forceXYZ]
            if extra_traces is not None:
                extra_data = tether.get_data(traces=extra_traces,
                                             samples=idx[0])
                data[c] = np.c_[data[c], extra_data]

    # Filter the data and plot the result
    resolution = tether.resolution
    if filter_time is None:
        filter_time = 0.005
    else:
        # filter_time has priority over filter_length
        filter_length_e = None
    window = window_var = max(int(np.round(filter_time * resolution)), 1)
    cap_data = True

    fbnl_filters = [[],[]]
    for c, cycle in enumerate(['stress', 'release']):  # 0=stress, 1=release
        if filter_length_e is not None:
            speed = _get_speed_approx(tether, i, cycle)
            filter_time = filter_length_e / speed  # s
            window = window_var = max(int(np.round(filter_time * resolution)), 1)
        for t in range(1, data[c].shape[1]):  # 1: extension, 2: force, 3: ...
            d = data[c][:, t]
            fbnl_filter = filter_fbnl(d, resolution, window=window,
                                      window_var=window_var, p=edginess,
                                      cap_data=cap_data)
            data[c][:, t] = fbnl_filter.data_filtered
            fbnl_filters[c].append(fbnl_filter)

    # Calculate angles with already filtered distance/force data
    if angles and angles_after_filter:
        for c in range(2):  # 1: stress, 2: release
            angle_extension = np.array([
                cart2sph(*point)[1:]
                for point in data[c][:, 7:10]])*180/math.pi
            angle_force = np.array([
                cart2sph(*point)[1:]
                for point in data[c][:, 10:13]])*180/math.pi
            if phi_shift_twopi:
                angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                angle_force[angle_force[:,1] < 0.0, 1] += 360
            data[c] = np.c_[data[c], angle_extension, angle_force]

    return data, fbnl_filters


def plot_force_extension(x, y, ystd=None, yerr=None, ax=None, show=False):
    if ax is None:
        # Create new figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Extension (nm)')
        ax.set_ylabel('Force (pN)')
        ax.set_title('Force Extension')

    # plot force extension lines and errorbars
    ax.plot(x * 1e9, y * 1e12)
    if ystd is not None:
        ax.errorbar(x * 1e9, y * 1e12, fmt='none', yerr=ystd * 1e12,
                    ecolor='grey', alpha=0.25)
    if yerr is not None:
        ax.errorbar(x * 1e9, y * 1e12, fmt='none', yerr=yerr * 1e12,
                    ecolor='black', alpha=0.25)

    if show:
        ax.get_figure().show()
    return ax


def _create_twin_ax(ax, subplot_pos=None):
    fig = ax.get_figure()
    subplot_pos = subplot_pos or (1, 1, 1)
    ax2 = fig.add_subplot(*subplot_pos, frame_on=False)
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    return ax2


def plot_angle_extension(x, theta_phi, axes=None, show=False):
    if axes is None:
        # Create new figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Apparent extension (nm)')
        ax.set_ylabel('Theta (°)')
        ax2 = _create_twin_ax(ax)
        ax2.set_ylabel('Phi (°)')
    else:
        ax, ax2 = axes

    # 0: theta_extension, 1: phi_exension, 2: theta_force, 3: phi_force
    lns1 = ax.plot(x * 1e9, theta_phi[:, 0], label=r'$\theta$ E')
    lns2 = ax.plot(x * 1e9, theta_phi[:, 2], label=r'$\theta$ F')
    lns3 = ax2.plot(x * 1e9, theta_phi[:, 1], label=r'$\phi$ E')
    lns4 = ax2.plot(x * 1e9, theta_phi[:, 3], label=r'$\phi$ F')

    lns = list(itertools.chain(lns1, lns2, lns3, lns4))
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    if show:
        ax.get_figure().show()

    return ax, ax2


def update_force_extension(tether, i=0, posmin=10e-9, bins=None,
                           resolution=None, sortcolumn=0,
                           ax=None, autoscale=True, xlim=None,
                           ylim=None):
    """
    Update the figure with force extension data.

    Parameters
    ----------
    xlim : (float, float), optional
        Set xlim of the axis.
    ylim : (float, float), optional
        Set ylim of the axis.
    """
    extra_traces = None
    # Calculate binned force extension data
    result = binned_force_extension(tether=tether, i=i, posmin=posmin,
                                    bins=bins, resolution=resolution,
                                    sortcolumn=sortcolumn,
                                    extra_traces=extra_traces)
    edges, centers, widths, bin_means, bin_stds, bin_Ns = result

    # time: 0, extension: 1, force: 2
    plotx = 1
    ploty = 2

    ax = ax or plt.gcf().gca()
    clear_force_extension(ax=ax)

    for d in range(2):  # 0 = stress, 1 = release
        e = bin_means[d][:, plotx]
        f = bin_means[d][:, ploty]
        fstd = bin_stds[d][:, ploty]
        ferr = bin_stds[d][:, ploty] / np.sqrt(bin_Ns[d])
        plot_force_extension(e, f, fstd, ferr, ax=ax)

    '''
    # Calculate force extension of a dna with a known length and plot it
    if bps:
        x, F = dna.force_extension(bps=bps)
        ax.lines[2].set_data(x*1e9, F*1e12)
    else:
        ax.lines[2].set_data([0], [0])
    '''

    if autoscale:
        ax.relim()
        # ax.autoscale_view()
        ax.autoscale()
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return ax


def clear_force_extension(ax=None):
    # clear old force extension lines and errorbars
    ax = ax or plt.gcf().gca()
    ax.set_prop_cycle(None)
    for l in ax.lines:
        l.remove()
    ax.lines.clear()
    for c in ax.containers:
        c.remove()
    ax.containers.clear()


def show_force_extension(tether, i=0, posmin=10e-9, bins=0, resolution=0,
                         sortcolumn=0, autoscale=False, xlim=None, ylim=None,
                         **kwargs):
    """
    Plot the force extension data with index `i` (see method
    `tether.force_extension_pairs()`) on tether.fe_figure.

    Parameters
    ----------
    i : int
        Index of force extension pair. See method
        `tether.force_extension_pair()`.
    xlim : (float, float), optional
        Xlimit of force extension axis.
    ylim : (float, float), optional
        Ylimit of force extension axis.
    """
    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Extension (nm)')
    ax.set_ylabel('Force (pN)')
    ax.set_title('Force Extension')

    def update_fe_pair(i, posmin, bins, resolution, sortcolumn, autoscale,
                       xlim_l, xlim_h, ylim_l, ylim_h):
        if bins <= 0:
            bins = None
        if resolution <= 0:
            resolution = None
        update_force_extension(tether, i, posmin=posmin, bins=bins,
                               resolution=resolution, sortcolumn=sortcolumn,
                               ax=ax, autoscale=autoscale,
                               xlim=(xlim_l, xlim_h),
                               ylim=(ylim_l, ylim_h), **kwargs)
        fig.canvas.draw()

    # Set default xlim/ylim values
    if xlim is None or ylim is None:
        _xlim, _ylim = autolimits(tether, posmin=posmin)
    xlim = xlim or _xlim
    ylim = ylim or _ylim

    # Get number of all force extension pairs
    stop = len(tether.stress_release_pairs(**kwargs)[0])

    # Build user interface (ui)
    index = BoundedIntText(value=i, min=0, max=stop - 1, description='FE_pair:')
    posmin = FloatText(value=posmin, description='PosMin')
    bins = IntText(value=bins, description='Bins')
    resolution = FloatText(value=resolution, step=1,
                           description='Resolution')
    sortcolumn = BoundedIntText(value=sortcolumn, min=0, max=2,
                                description='Sortcolumn')
    autolim = Checkbox(value=autoscale, description='Autoscale')
    xlim_l = FloatText(value=xlim[0])
    xlim_h = FloatText(value=xlim[1])
    ylim_l = FloatText(value=ylim[0])
    ylim_h = FloatText(value=ylim[1])
    xlim_b = HBox((Label('xlim'), xlim_l, xlim_h))
    ylim_b = HBox((Label('ylim'), ylim_l, ylim_h))
    ui_fe = VBox((HBox((index, posmin)), HBox((bins, resolution, sortcolumn))))
    ui_plot = VBox((autolim, xlim_b, ylim_b))

    # initialize force extension plot
    update_fe_pair(i, posmin.value, bins.value, resolution.value,
                   sortcolumn.value, autolim.value, xlim[0], xlim[1], ylim[0],
                   ylim[1])

    # Make user input fields interactive
    out = interactive_output(update_fe_pair, {'i': index,
                                              'posmin': posmin,
                                              'bins': bins,
                                              'resolution': resolution,
                                              'sortcolumn': sortcolumn,
                                              'autoscale': autolim,
                                              'xlim_l': xlim_l,
                                              'xlim_h': xlim_h,
                                              'ylim_l': ylim_l,
                                              'ylim_h': ylim_h})

    # Show user interface
    display(ui_fe)
    fig.show()
    display(ui_plot)

    return ui_fe, fig, ui_plot


def autolimits(tether, posmin=10e-9, samples=None, e=None, f=None, xlim=None,
               ylim=None):
    """
    Determine xlim and ylim values for force extension plots.

    Parameters
    ----------
    samples : int, slice or index array, optional
        Samples to get extension and force from.
    e : 1D numpy.ndarray of floats, optional
        Extension in nm. Takes precedence over extension determined with
        `samples`.
    f : 1D numpy.ndarray of floats, optional
        Force in pN. Takes precedence over force determined with `samples`.
    xlim : (float, float), optional
        Xlimit of force extension axis. Takes precedence over xlim
        determined with `e`.
    ylim : (float, float), optional
        Ylimit of force extension axis. Takes precedence over ylim
        determined with `f`.

    Returns
    -------
    (float, float)
        The xlim
    (float, float)
        The ylim
    """
    if samples is None \
            and (xlim is None and e is None) \
            or (ylim is None and f is None):
        # Get the start/stop indices of the data to be used to determine
        # the min max values
        sts, rls = tether.stress_release_pairs()
        start = sts[0].start
        stop = rls[-1].stop
        samples = slice(start, stop)

    if xlim is None and ylim is None and e is None and f is None:
        e_f = tether.force_extension(samples=samples, posmin=posmin)  # m, N
        e = e_f[:, 0]
        f = e_f[:, 1]
    if xlim is None and e is None:
        e = tether.extension(samples=samples, posmin=posmin) # m
    if ylim is None and f is None:
        f = tether.force(samples=samples, posmin=posmin) # N

    if xlim is None:
        e_min = e.min()
        e_max = e.max()
        e_diff = (e_max - e_min) * 0.02
        xlim = ((e_min - e_diff) * 1e9, (e_max + e_diff) * 1e9)

    if ylim is None:
        f_min = f.min()
        f_max = f.max()
        f_diff = (f_max - f_min) * 0.02
        ylim = ((f_min - f_diff) * 1e12, (f_max + f_diff) * 1e12)

    # Return the set limits
    return xlim, ylim


def save_figures(figures, directory=None, file_prefix=None, file_suffix=None,
                 file_extension='.png', index_digits=3):
    """
    Save matplotlib figures in a given directory.

    The filenames of the figures will be a concatenation of the `file_prefix`,
    an index with `index_digits` digits, the `file_suffix` and the
    `file_extension`.

    Parameters
    ----------
    figures : Iterable of matplotlib figures or one figure
        A list, array, generator or other Iterable type of matplotlib figures.
        If figures is only one matplotlib figure, no index will be included in
        the filename of the figure.
    directory : str
        The directory, the figures should be saved in. The directory will be
        created, if it does not exist.
    file_prefix : str, optional
        A prefix every filename of the saved figures should include.
    file_suffix : str, optional
        A suffix every filename of the saved figures should include.
    files_extension : str, optional
        The file extension (and type) to be used to save the figures (default
        '.png').
    index_digits : int, optional
        Digits to be used for the index in the filename of the figures.
    """
    directory = directory or os.path.join(".", "results")
    file_prefix = file_prefix or ""
    file_suffix = file_suffix or ""

    # create results dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    # If only one figure, save it without an index
    if not isinstance(figures, collections.Iterable):
        filename = "%s%s%s" % (file_prefix, file_suffix, file_extension)
        figures.savefig(os.path.join(directory, filename))

    for idx, fig in enumerate(figures):
        format_string = "".join(("%s%.", str(index_digits), "i%s%s"))
        filename = format_string % (file_prefix, idx, file_suffix,
                                    file_extension)
        fig.savefig(os.path.join(directory, filename))

'''
    def save_force_extension_plots(self, directory=None, file_prefix=None,
                                   file_suffix=None, file_extension='.png',
                                   **kwargs):
        """
        Save all plots created by `plot_force_extensions()`.

        directory : str
            The directory the images to be displayed are located in.
        file_prefix : str
            Display only the files beginning with `prefix`.
        file_suffix : str
            Display only the files ending with `suffix`.
        file_extension : str, optional
            The extension of the images that should be displayed. Default is
            '.png'.
        figure : matplotlib figure, optional
            A reference to a figure that should be used to plot the force
            extension pairs. If no figure is given, a new one is automatically
            created.
        **kwargs
            Parameters passed to the method `self.plot_force_extensions()`.
        """
        kwargs.pop('draw', None)
        # Create generator for all force/extension stress/release pairs
        figures = self.plot_force_extensions(draw=False, **kwargs)

        # Save all figures
        evaluate.save_figures(figures, directory=directory,
                              file_prefix=file_prefix, file_suffix=file_suffix,
                              file_extension=file_extension)

        # Redraw the figure, after the last one has been saved
        self.fe_figure.canvas.draw()
'''


def get_simulation(tether, i, settings_file, posZ=None, individual_posZ=False,
                   kappa=None, kappa_z_factor=None, excited_axis=None,
                   **kwargs):
    """
    Get unzipping simulation for tether force extension segment number `i`.

    Determine `kappa` and `positionZ` from `tether` to get a proper simulation.

    Parameters
    ----------
    tether : Tether
        The tether object
    i : int
        The segment number to get the unzipping simulation for
    settings_file : str
        The filepath of the settings file for the simulation
    individual_posZ : bool
        Calculate the median of the distance of the microsphere to the surface
        from positionZ for each individual segment or the whole tether.region.
    posZ : float
        Set the positionZ manually (m).
    """
    # Get radius from calibration
    radius = tether.calibration.radius

    # Determine distance between microsphere and surface
    if posZ is None:
        idx = None
        if individual_posZ:
            idx = tether.samples(i, cycle='stress')
        posZ = np.median(tether.get_data('positionZ', samples=idx))
        h0 = max(0.0, -posZ * tether.calibration.focalshift)

    # Get kappa for excited axis and axis Z
    kappa = tether.calibration.kappa(posZ) if kappa is None else kappa
    kappa_z_factor = 1 if kappa_z_factor is None else kappa_z_factor
    if excited_axis is None:
        axis = {'x': 0, 'y': 1}
        ax = tether.stress_release_pairs(i=i, info=True)[2][0,0]
        excited_axis = axis[ax]
    axes_kappa = [excited_axis, 2]
    kappa = kappa[axes_kappa] * np.array([1, kappa_z_factor])

    # Get/do simulation with simulation_settings_file and radius, h0, and kappa
    simulation = uzsi.get_unzipping_simulation(settings_file, radius=radius,
                                               kappa=kappa, h0=h0, **kwargs)

    return simulation


def plot_unzip_data(tether, I, ax=None, fbnl=False, shift_x=0e-9, t_delta=15,
                    plot_stress=True, plot_release=True, plot_raw=False,
                    annotate_stress=True, annotate_release=True,
                    simulation=None, **filter_kwargs):
    """
    i : int or list of ints
        the force extension data to be plotted
    t_delta : float
        Time in seconds the microsphere was trapped before the start of the very
        first stress-release cycle
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ###############################
    ### Get and plot simulated data
    if simulation is not None:
        e_sim, f_sim, fXYZ, nuz = uzsi.get_force_extension_nuz(simulation,
                                                               theta=False)
        # Apparent extension and average force acting on the microsphere
        ax.plot(e_sim[f_sim<=25e-12]*1e9, f_sim[f_sim<=25e-12]*1e12,
                color='#000000', ls=(0, (2.5, 2.5)), label='Simulation')
                # (offset, (on_off_seq)) dashes=(3, 2))
        #ax.annotate('Simulation', xy=(650, 7), xycoords='data', color=c)
        #            # xytext=(21.35, -15), textcoords='offset points',
        #            # arrowprops=dict(linewidth=1.25, arrowstyle="->", color=c))

    if not isinstance(I, collections.Iterable):
        I = [I]
    for i in I:
        ##################################
        ### Get and plot raw force extension
        if plot_raw:
            # Get raw extension and force for stress and release cycle
            fe_pair = tether.force_extension_pair(i=i, time=True)
            (x_stress, f_stress, info_stress,
             x_release, f_release, info_release,
             t_stress, t_release) = fe_pair
            extension_raw = np.r_[x_stress, x_release]
            force_raw = np.r_[f_stress, f_release]
            ax.plot((extension_raw + shift_x) * 1e9, force_raw * 1e12, c='#CCCCCC')

        ###############################################################################
        ### Get and plot force extension curve and simulation
        if fbnl:
            # Get fbnl_filter filtered force extension
            result = fbnl_force_extension(tether, i, **filter_kwargs)
            filtered_data, fbnl_filters = result  # 0: stress, 1: release
        else:
            # Get binned force extension
            bin_edges, bin_centers, bin_widths, bin_means, bin_stds, bin_Ns \
                = binned_force_extension(tether, i, **filter_kwargs)
            filtered_data = bin_means  # 0: stress, 1: release

        # Get tmin and tmax from the first stress and the last release cycle datapoint
        tmin = filtered_data[0][0,0] + t_delta
        tmax = filtered_data[1][-1,0] + t_delta
        time = '$t={:3.0f}-{:3.0f}\,s$'.format(tmin, tmax)
        time = '$t={:3.0f}\,s$'.format((tmin+tmax)/2)

        # Plot release cycle [1]
        if plot_release:
            cycle = 1
            pre = 'rls '
            c = 'magenta'
            ax.plot((filtered_data[cycle][:,1] + shift_x) * 1e9,
                    filtered_data[cycle][:,2] * 1e12,
                    label='{}{}'.format(pre, time))
            if annotate_release:
                ax.annotate('Release', xy=(700, 23), xycoords='data', color=c)
                            # xytext=(5, -30), textcoords='offset points',
                            # arrowprops=dict(linewidth=1.25, arrowstyle="->", color=c))
                ax.annotate("", xytext=(700, 22), xy=(750, 22), xycoords='data',
                            arrowprops=dict(linewidth=1.25, arrowstyle="<-", color=c))

        # Plot stress cycle [0]
        if plot_stress:
            cycle = 0
            pre = 'str '
            c = 'cyan'
            ax.plot((filtered_data[cycle][:,1] + shift_x) * 1e9,
                    filtered_data[cycle][:,2] * 1e12,
                    label='{}{}'.format(pre, time))
            if annotate_stress:
                ax.annotate('Stretch', xy=(600, 23), xycoords='data', color=c)
                            # xytext=(5, 20), textcoords='offset points',
                            # arrowprops=dict(linewidth=1.25, arrowstyle="->", color=c))
                ax.annotate("", xytext=(600, 22), xy=(650, 22), xycoords='data',
                            arrowprops=dict(linewidth=1.25, arrowstyle="->", color=c))

    ax.set_xlabel('Extension (nm)')
    ax.set_ylabel('Force (pN)')

    return fig, ax
