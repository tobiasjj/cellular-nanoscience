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

from matplotlib import pyplot as plt
from IPython.display import display
from ipywidgets import Label, interactive_output, Checkbox, IntText, BoundedIntText, FloatText, HBox, VBox
from .binning import calculate_bin_means
import math
import numpy as np
import os
import collections
import itertools
from ...stepfinder.stepfinder import filter_fbnl


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


def binned_force_extension(tether, i, bins=None, resolution=None, sortcolumn=0,
                           angles=False, extra_traces=None,
                           angles_after_binning=False, avoid_twopi_switch=False):
    """
    Parameters
    ----------
    bins : int or str
        number of bins, takes precedence over resolution
    resolution : float
        number of bins per unit of sortcolumn
    sortcolumn : int
        0: time, 1: extension, 2: force, n >= 3: angles and/or extra_columns
    angles : bool
        Calculate theta and phi for extension and force.
        3: theta_extension, 4: phi_extension, 5: theta_force, 6: phi_force,
        7,8,9: distanceXYZ, 10,11,12: forceXYZ
    angles_after_binning : bool
        13,14,15,16: theta (13,15) and phi (14,16) for extension and force
    """
    fe_pair = tether.force_extension_pair(i=i, time=True)
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
        for i, idx in enumerate(idxs):  # 1: stress, 2: release
            if angles:
                # Get distance/force (vectors XYZ)
                distanceXYZ = tether.distanceXYZ(samples=idx[0])
                forceXYZ = tether.forceXYZ(samples=idx[0])
                # calculate angles theta and phi
                angle_extension = np.array([
                        cart2sph(*point)[1:] for point in distanceXYZ
                        ])*180/math.pi
                angle_force = np.array([
                        cart2sph(*point)[1:] for point in forceXYZ
                        ])*180/math.pi
                if avoid_twopi_switch:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                data[i] = np.c_[data[i], angle_extension, angle_force,
                                distanceXYZ, forceXYZ]
            if extra_traces is not None:
                extra_data = tether.get_data(traces=extra_traces,
                                             samples=idx[0])
                data[i] = np.c_[data[i], extra_data]

    edges = [None]*2
    centers = [None]*2
    widths = [None]*2
    bin_means = [None]*2
    bin_stds = [None]*2
    bin_Ns = [None]*2
    for i in range(2):  # 0=stress, 1=release
        result = calculate_bin_means(data[i], bins=bins, resolution=resolution,
                                     sortcolumn=sortcolumn)
        (edges[i], centers[i], widths[i],
         bin_means[i], bin_stds[i], bin_Ns[i]) = result

    # Calculate angles with already binned distance/force data
    if angles and angles_after_binning:
            for i in range(2):  # 1: stress, 2: release
                angle_extension = np.array([
                        cart2sph(*point)[1:]
                        for point in bin_means[i][:, 7:10]])*180/math.pi
                angle_force = np.array([
                        cart2sph(*point)[1:]
                        for point in bin_means[i][:, 10:13]])*180/math.pi
                if avoid_twopi_switch:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                bin_means[i] = np.c_[bin_means[i], angle_extension, angle_force]

    return edges, centers, widths, bin_means, bin_stds, bin_Ns


def fbnl_force_extension(tether, i, filter_time=0.005, edginess=1,
                         angles=False, extra_traces=None,
                         angles_after_filter=False, avoid_twopi_switch=False):
    """
    Parameters
    ----------
    bins : int or str
        number of bins, takes precedence over resolution
    resolution : float
        number of bins per unit of sortcolumn
    sortcolumn : int
        0: time, 1: extension, 2: force, n >= 3: angles and/or extra_columns
    angles : bool
        Calculate theta and phi for extension and force.
        3: theta_extension, 4: phi_extension, 5: theta_force, 6: phi_force,
        7,8,9: distanceXYZ, 10,11,12: forceXYZ
    angles_after_filter : bool
        13,14,15,16: theta (13,15) and phi (14,16) for extension and force
    """
    fe_pair = tether.force_extension_pair(i=i, time=True)
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
        for i, idx in enumerate(idxs):  # 1: stress, 2: release
            if angles:
                # Get distance/force (vectors XYZ)
                distanceXYZ = tether.distanceXYZ(samples=idx[0])
                forceXYZ = tether.forceXYZ(samples=idx[0])
                # calculate angles theta and phi
                angle_extension = np.array([
                        cart2sph(*point)[1:] for point in distanceXYZ
                        ])*180/math.pi
                angle_force = np.array([
                        cart2sph(*point)[1:] for point in forceXYZ
                        ])*180/math.pi
                if avoid_twopi_switch:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                data[i] = np.c_[data[i], angle_extension, angle_force,
                                distanceXYZ, forceXYZ]
            if extra_traces is not None:
                extra_data = tether.get_data(traces=extra_traces,
                                             samples=idx[0])
                data[i] = np.c_[data[i], extra_data]

    # Filter the data and plot the result
    resolution = tether.resolution
    window = window_var = max(int(np.round(filter_time * resolution)), 1)
    cap_data = True

    fbnl_filters = [[None]]*2
    for i in range(2):  # 0=stress, 1=release
        for c in range(1, data[i].shape[1]):  # 1: extension, 2: force, 3: ...
            d = data[i][:, c]
            fbnl_filter = filter_fbnl(d, resolution, window=window,
                                      window_var=window_var, p=edginess,
                                      cap_data=cap_data)
            data[i][:, c] = fbnl_filter.data_filtered
            fbnl_filters[i].append(fbnl_filter)

    # Calculate angles with already filtered distance/force data
    if angles and angles_after_filter:
            for i in range(2):  # 1: stress, 2: release
                angle_extension = np.array([
                        cart2sph(*point)[1:]
                        for point in data[i][:, 7:10]])*180/math.pi
                angle_force = np.array([
                        cart2sph(*point)[1:]
                        for point in data[i][:, 10:13]])*180/math.pi
                if avoid_twopi_switch:
                    angle_extension[angle_extension[:,1] < 0.0, 1] += 360
                    angle_force[angle_force[:,1] < 0.0, 1] += 360
                data[i] = np.c_[data[i], angle_extension, angle_force]

    return data, fbnl_filters


def plot_force_extension(x, y, ystd=None, yerr=None, ax=None, show=False):
    if ax is None:
        # Create new figure
        fig, ax = plt.subplots()
        ax.set_xlabel('Extension (nm)')
        ax.set_ylabel('Force (pN)')
        ax.set_title('Force Extension')

    # plot force extension lines and errorbars
    ax.plot(x, y, alpha=0.99)
    if ystd is not None:
        ax.errorbar(x, y, fmt='none', yerr=ystd, ecolor='grey', alpha=0.25)
    if yerr is not None:
        ax.errorbar(x, y, fmt='none', yerr=yerr, ecolor='black', alpha=0.25)

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
    lns1 = ax.plot(x, theta_phi[:, 0], label=r'$\theta$ E')
    lns2 = ax.plot(x, theta_phi[:, 2], label=r'$\theta$ F')
    lns3 = ax2.plot(x, theta_phi[:, 1], label=r'$\phi$ E')
    lns4 = ax2.plot(x, theta_phi[:, 3], label=r'$\phi$ F')

    lns = list(itertools.chain(lns1, lns2, lns3, lns4))
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    if show:
        ax.get_figure().show()

    return ax, ax2


def update_force_extension(tether, i=0, bins=None, resolution=None,
                           sortcolumn=0,
                           ax=None, autoscale=True, xlim=None,
                           ylim=None, extra_traces=None):
    """
    Update the figure with force extension data.

    Parameters
    ----------
    xlim : (float, float), optional
        Set xlim of the axis.
    ylim : (float, float), optional
        Set ylim of the axis.
    """
    # Calculate binned force extension data
    result = binned_force_extension(tether=tether, i=i, bins=bins,
                                    resolution=resolution,
                                    sortcolumn=sortcolumn,
                                    extra_traces=extra_traces)
    edges, centers, widths, bin_means, bin_stds, bin_Ns = result

    # time: 0, extension: 1, force: 2
    plotx = 1
    ploty = 2

    ax = ax or plt.gcf().gca()

    clear_force_extension(ax=ax)

    for d in range(2):  # 0 = stress, 1 = release
        plot_force_extension(
            bin_means[d][:, plotx],
            bin_means[d][:, ploty],
            bin_stds[d][:, ploty],
            bin_stds[d][:, ploty] / np.sqrt(bin_Ns[d]),
            ax=ax)

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


def show_force_extension(tether, i=0, bins=0, resolution=0, sortcolumn=0,
                         autoscale=False, xlim=None, ylim=None, **kwargs):
    """
    Plot the force extension data with index `i` (see method
    `self.force_extension_pairs()`) on self.fe_figure.
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

    def update_fe_pair(i, bins, resolution, sortcolumn, autoscale,
                       xlim_l, xlim_h, ylim_l, ylim_h):
        if bins <= 0:
            bins = None
        if resolution <= 0:
            resolution = None
        update_force_extension(tether, i, bins=bins, resolution=resolution,
                               sortcolumn=sortcolumn,
                               ax=ax, autoscale=autoscale,
                               xlim=(xlim_l, xlim_h),
                               ylim=(ylim_l, ylim_h), **kwargs)
        fig.canvas.draw()

    # Set default xlim/ylim values
    if xlim is None or ylim is None:
        _xlim, _ylim = autolimits(tether)
    xlim = xlim or _xlim
    ylim = ylim or _ylim

    # Get number of all force extension pairs
    stop = len(tether.stress_release_pairs(**kwargs)[0])

    # Build user interface (ui)
    index = BoundedIntText(value=i, min=0, max=stop - 1, description='FE_pair:')
    bins = IntText(value=bins, description='Bins')
    resolution = FloatText(value=resolution, step=0.1,
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
    ui_fe = VBox((index, HBox((bins, resolution, sortcolumn))))
    ui_plot = VBox((autolim, xlim_b, ylim_b))

    # initialize force extension plot
    update_fe_pair(i, bins.value, resolution.value, sortcolumn.value,
                   autolim.value, xlim[0], xlim[1], ylim[0], ylim[1])

    # Make user input fields interactive
    out = interactive_output(update_fe_pair, {'i': index,
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


def autolimits(tether, samples=None, e=None, f=None, xlim=None, ylim=None):
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
                e_f = tether.force_extension(samples=samples) * 1000  # nm, pN
                e = e_f[:, 0]
                f = e_f[:, 1]
        if xlim is None and e is None:
            e = tether.extension(samples=samples) * 1000  # nm
        if ylim is None and f is None:
            f = tether.force(samples=samples) * 1000  # pN

        if xlim is None:
            e_min = e.min()
            e_max = e.max()
            e_diff = (e_max - e_min) * 0.02
            xlim = (e_min - e_diff, e_max + e_diff)

        if ylim is None:
            f_min = f.min()
            f_max = f.max()
            f_diff = (f_max - f_min) * 0.02
            ylim = (f_min - f_diff, f_max + f_diff)

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
