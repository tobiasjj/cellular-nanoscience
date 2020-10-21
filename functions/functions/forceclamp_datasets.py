import os
import numpy as np
import pyoti
import stepfinder as sf

from matplotlib import pyplot as plt

from .filtering import filter_fbnl_data, filter_fbnl_region
from .helpers import suppress_stdout

DATA_BASEDIR = './data'
RESULTS_REGION_NAME = 'results'
RESOLUTION_SF = 1000


def get_dbfile_path(dataset, basedir=None):
    basedir = '.' if basedir is None else basedir
    directory = dataset['directory']
    dbname = dataset['dbname']
    fullpath = os.path.join(basedir, directory, dbname + '.fs')
    return fullpath


def get_forceclamp_data(dataset, deactivate_baseline_mod=False,
                        baseline_decimate=1):
    results_region_name = RESULTS_REGION_NAME
    feedback_axes = dataset['feedback_axes']
    sign_axes = dataset.get('sign_axes', [None]*len(feedback_axes))
    tmins = dataset['tmins']
    tmaxs = dataset['tmaxs']
    limits = dataset['limits']

    # Open experiment
    dbfile = get_dbfile_path(dataset, DATA_BASEDIR)
    experiment = pyoti.open_experiment(filename=dbfile,
                                       return_last_created=False)
    try:
        mod = experiment.modification('baseline_mod')
        if deactivate_baseline_mod:
            mod.active = False
        else:
            mod.iattributes.baseline_decimate = baseline_decimate
    except:
        pass

    with suppress_stdout():
        experiment.set_cached_region(results_region_name)

    data = {}
    try:
        for i, (fa, sas, tmin, tmax, limit) in enumerate(zip(feedback_axes,
                                                             sign_axes, tmins,
                                                             tmaxs, limits)):
            d = _get_forceclamp_data(dataset, experiment, results_region_name,
                                     fa, sas, tmin, tmax)
            d['key'] = dataset['key']
            d['limit'] = limit
            d['bin_resolution'] = dataset['bin_resolution']
            # d['befores'] = dataset['befores']
            # d['afters'] = dataset['afters']
            data[i] = d
    finally:
        # Abort changes and close experiment
        experiment.close(abort=True)

    return data


def _get_forceclamp_data(dataset, experiment, results_region_name,
                         feedback_axis, sign_axes, tmin, tmax):
    traces = dataset['traces']
    trace_factors = dataset.get('trace_factors', None)
    fa = feedback_axis
    posmin = dataset.get('posmin', 1e-8)

    # Get and filter feedback
    filter_time = 0.1  # s
    edginess = 1
    data, data_filtered, fbnl_filters = \
        filter_fbnl_region(experiment, results_region_name, traces, time=True,
                           tmin=tmin, tmax=tmax, filter_time=filter_time,
                           edginess=edginess)
    # Apply trace factors and subtract the mirror offset
    if trace_factors is not None:
        for j, trace_factor in enumerate(trace_factors):
            data[:,j+1] *= trace_factor
            data_filtered[:,j+1] *= trace_factor
    offset_feedback = np.array([-5.0045e-6, -5.0085e-6])
    data[:,1:] -= offset_feedback
    data_filtered[:,1:] -= offset_feedback

    # Get and filter the extension, corrected with feedback
    results_region = experiment.region(results_region_name)
    resolution = results_region.samplingrate
    start = int(round(tmin*resolution))
    stop = int(round(tmax*resolution)) + 1
    samples = slice(start, stop)
    _data = results_region.get_data(['psdXYZ', 'positionXYZ'], samples=samples)
    psdXYZ = _data[:,:3]
    positionXYZ = _data[:,3:]
    positionXY = _data[:,3:5]
    positionZ = _data[:,5]
    positionXY_feedback = positionXY - data[:,1:]
    positionXYZ_feedback = np.c_[positionXY_feedback, positionZ]
    distanceXYZ = \
        pyoti.evaluate.tether.distanceXYZ(psdXYZ, positionXYZ_feedback,
                                        calibration=results_region.calibration)
    distance = pyoti.evaluate.tether.distance(distanceXYZ, positionXY_feedback,
                                              posmin=posmin)
    extension = \
        pyoti.evaluate.tether.extension(distance,
                                        results_region.calibration.radius)
    extension_filtered, fbnl_extension_filters = \
        filter_fbnl_data(extension, resolution, filter_time=filter_time,
                         edginess=edginess)

    # Get and filter the force
    forceXYZ = pyoti.evaluate.tether.forceXYZ(psdXYZ,
                                        calibration=results_region.calibration,
                                        positionZ=positionZ)
    force = pyoti.evaluate.tether.force(forceXYZ, positionXY_feedback,
                                        posmin=posmin, sign_axes=sign_axes)
    force_filtered, fbnl_force_filters = \
        filter_fbnl_data(force, resolution, filter_time=filter_time,
                         edginess=edginess)

    _data = {}
    _data['time'] = data[:,0]
    _data['feedback'] = data[:,fa]
    _data['extension'] = extension
    _data['force'] = force
    _data['feedback_filtered'] = data_filtered[:,fa]
    _data['extension_filtered'] = extension_filtered[:,0]
    _data['force_filtered'] = force_filtered[:,0]
    _data['samplingrate'] = results_region.samplingrate
    return _data


def find_steps(forceclamp_data, steps_trace='extension',
               expected_min_step_size=1e-9, filter_time=None,
               filter_min_t=None, filter_max_t=None, filter_number=None,
               edginess=1, expected_min_dwell_t=None,
               step_size_threshold=None):
    # get the data
    resolution = forceclamp_data['samplingrate']  # in Hz
    data = forceclamp_data[steps_trace]

    step_finder_result \
        = sf.filter_find_analyse_steps(data, resolution, filter_time,
                                       filter_min_t, filter_max_t,
                                       filter_number, edginess,
                                       expected_min_step_size,
                                       expected_min_dwell_t,
                                       step_size_threshold, pad_data=True,
                                       verbose=False, plot=False)
    return step_finder_result


def plot_force_feedback(t, force, force_filtered, feedback, feedback_filtered,
                        ax=None, xlim=None, ylim=None, xlabel=True,
                        xticklabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot the raw data of force and feedback
    ax.plot(t, force*1e12, alpha=0.05, c='black')
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    ax2.plot(t, feedback*1e9, alpha=0.2, c='black')

    # Plot the force
    line, = ax.plot(t, force_filtered*1e12)
    c = line.get_color()
    ax.yaxis.label.set_color(c)
    ax.tick_params(axis='y', colors=c)
    if xlabel:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (pN)')

    # Plot the feedback
    line, = ax2.plot(t, feedback_filtered*1e9)
    c = line.get_color()
    ax2.yaxis.label.set_color(c)
    ax2.tick_params(axis='y', colors=c)
    ax2.set_ylabel('Feedback (nm)')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, (ax, ax2)


def plot_force_extension(t, force, force_filtered, extension,
                         extension_filtered, ax=None, xlim=None, ylim=None,
                         xlabel=True, xticklabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot the raw data of force and extension
    ax.plot(t, force*1e12, alpha=0.05, c='black')
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    ax2.plot(t, extension*1e9, alpha=0.2, c='black')

    # Plot the force
    line, = ax.plot(t, force_filtered*1e12)
    c = line.get_color()
    ax.yaxis.label.set_color(c)
    ax.tick_params(axis='y', colors=c)
    if xlabel:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (pN)')

    # Plot the extension
    line, = ax2.plot(t, extension_filtered*1e9)
    c = line.get_color()
    ax2.yaxis.label.set_color(c)
    ax2.tick_params(axis='y', colors=c)
    ax2.set_ylabel('Extension (nm)')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, (ax, ax2)


def plot_feedback_histogram(feedback, bins, ax=None, xlim=None, ylim=None,
                            xlabel=True, xticklabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # plot the histogram of the feedback
    ax.hist(feedback*1e9, bins=bins, orientation='vertical')
    # ax.set_yscale('log')
    if xlabel:
        ax.set_xlabel('Feedback (nm)')
    ax.set_ylabel('Number')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax


def plot_feedback_histogram_fft(feedback, bins, ax=None, xlim=None, ylim=None,
                                xlabel=True, xticklabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # calculate the fft of the feedback histogram to detect repeating steps
    numbers, edges = np.histogram(feedback*1e9, bins=bins)
    step = edges[1] - edges[0]
    extent = edges[-1] - edges[0]
    frequencies = np.fft.rfft(numbers)
    ax.plot(extent / np.arange(1, len(frequencies)), np.abs(frequencies[1:]))
    if xlabel:
        ax.set_xlabel('Stepsize (nm)')
    ax.set_ylabel('Amplitude (#)')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax


def plot_extension_histogram(extension, bins, ax=None, xlim=None, ylim=None,
                             xlabel=True, xticklabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # plot the histogram of the extension
    ax.hist(extension*1e9, bins=bins, orientation='vertical')
    # ax.set_yscale('log')
    if xlabel:
        ax.set_xlabel('Extension (nm)')
    ax.set_ylabel('Number')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax


def plot_forceclamp_data(forceclamp_data, forceclamp_figs_dir=None, show=False):
    save = False if forceclamp_figs_dir is None else True
    for i, data in forceclamp_data.items():
        t = data['time']
        feedback = data['feedback']
        extension = data['extension']
        force = data['force']
        feedback_filtered = data['feedback_filtered']
        extension_filtered = data['extension_filtered']
        force_filtered = data['force_filtered']
        key = data['key']
        limit = data['limit']
        bin_resolution = data['bin_resolution']

        # Plot the feedback
        fig, (ax, ax2) = plot_force_feedback(t, force, force_filtered,
                                             feedback, feedback_filtered)
        title = 'Feedback Position {} {:02d}'.format(key, i + 1)
        ax.set_title(title)
        if save:
            try:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.png'.format(title)))
            except:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.pdf'.format(title)))
        if show: fig.show()
        else: plt.close(fig)

        # Plot the extension
        fig, (ax, ax2) = plot_force_extension(t, force, force_filtered,
                                              extension, extension_filtered)
        title = 'Extension {} {:02d}'.format(key, i + 1)
        ax.set_title(title)
        if save:
            try:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.png'.format(title)))
            except:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.pdf'.format(title)))
        if show: fig.show()
        else: plt.close(fig)

        for n, (op, lim) in enumerate(limit):
            idx = (op(feedback*1e9, lim)).flatten()

            # plot the histogram of the feedback
            bins = int(round((np.max(feedback[idx])
                              - np.min(feedback[idx]))*1e9)) * bin_resolution
            fig, ax = plot_feedback_histogram(feedback[idx], bins)
            title = 'Feedback Histogram {} {:02d}_{}'.format(key, i + 1, n + 1)
            ax.set_title(title)
            if save:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.png'.format(title)))
            if show: fig.show()
            else: plt.close(fig)

            # calculate the fft of the feedback histogram to detect repeating
            # steps
            fig, ax = plot_feedback_histogram_fft(feedback[idx], bins,
                                                  xlim=(0, 70))
            title = 'Steps in Histogram {} {:02d}_{}'.format(key, i + 1, n + 1)
            ax.set_title(title)
            if save:
                fig.savefig(os.path.join(forceclamp_figs_dir, '{}.png'.format(title)),
                            transparent=False)
            if show: fig.show()
            else: plt.close(fig)

            # plot the histogram of the extension
            bins = int(round((np.max(extension[idx])
                              - np.min(extension[idx]))*1e9)) * bin_resolution
            fig, ax = plot_extension_histogram(extension[idx], bins)
            title = 'Extension Histogram {} {:02d}_{}'.format(key, i + 1,
                                                              n + 1)
            ax.set_title(title)
            if save:
                fig.savefig(os.path.join(forceclamp_figs_dir,
                                         '{}.png'.format(title)))
            if show: fig.show()
            else: plt.close(fig)

    #for i, (before, after) in enumerate(zip(befores, afters)):
    #    out[0].children[0].value = before
    #    title = 'Force Extension Before {} {:02d}'.format(key, i+1)
    #    plt.gcf().gca().set_title(title)
    #    plt.gcf().canvas.draw()
    #    plt.savefig(os.path.join(path, '{}.png'.format(title)),
    #                transparent=False)

    #    out[0].children[0].value = after
    #    title = 'Force Extension After {} {:02d}'.format(key, i+1)
    #    plt.gcf().gca().set_title(title)
    #    plt.gcf().canvas.draw()
    #    plt.savefig(os.path.join(path, '{}.png'.format(title)),
    #                transparent=False)
