import itertools
import numpy as np
import os
import pyoti
import pickle
import scipy.signal
import unzipping_simulation as uzsi
import warnings

from collections.abc import Iterable
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import simps, cumtrapz

from .binning import calculate_bin_means, concatenate_data_dict, separate_data_array
from .helpers import compare_idx, min_max_idx, step_idx
from .force_extension import binned_force_extension, fbnl_force_extension, \
                             get_simulation
from .helpers import suppress_stdout

DATA_BASEDIR = './data'
RESULTS_REGION_NAME = 'results'
RESOLUTION_SF = 1000
SIMULATION_SETTINGS_FILE = './simulation_settings.p'
SIMULATIONS_DIR = './simulations'
CYCLES_DATAS_DIR = './cycles_datas'
CYCLES_MEANS_DIR = './cycles_means'

def create_dataset(directory, dbname, kappa_z_factor, shift_x, c_rad52=0,
                   c_count=0, number_of_pairs=None, key=None, datadir='data'):
    directory = os.path.join(directory, datadir)
    dataset = {
        'key': key,
        'directory': directory,
        'dbname': dbname,
        'kappa_z_factor': kappa_z_factor,
        'shift_x': shift_x,
        'c_rad52': c_rad52,
        'c_count': c_count,
        'number_of_pairs': number_of_pairs,
    }
    return dataset


def get_dbfile_path(dataset, basedir=None):
    basedir = '.' if basedir is None else basedir
    directory = dataset['directory']
    dbname = dataset['dbname']
    fullpath = os.path.join(basedir, directory, dbname + '.fs')
    return fullpath


def get_cycles_data_filepath(dataset, cycles_datas_dir=None, makedirs=False):
    cycles_datas_dir = CYCLES_DATAS_DIR if cycles_datas_dir is None \
                       else cycles_datas_dir
    filename = '{}.p'.format(dataset['key'])
    filepath = os.path.join(cycles_datas_dir, filename)
    if makedirs:
        os.makedirs(cycles_datas_dir, exist_ok=True)
    return filepath

def get_cycles_mean_filepath(dataset, cycles_means_dir=None, makedirs=False):
    cycles_means_dir = CYCLES_MEANS_DIR if cycles_means_dir is None \
                       else cycles_means_dir
    return get_cycles_data_filepath(dataset, cycles_datas_dir=cycles_means_dir,
                                    makedirs=makedirs)

def load_cycles_data(dataset, cycles_datas_dir=None):
    filepath = get_cycles_data_filepath(dataset,
                                        cycles_datas_dir=cycles_datas_dir)
    try:
        with open(filepath, 'rb') as f:
            cycles_datas = pickle.load(f)
    except FileNotFoundError:
        cycles_datas = None
    return cycles_datas

def load_cycles_mean(dataset, cycles_means_dir=None):
    cycles_means_dir = CYCLES_MEANS_DIR if cycles_means_dir is None \
                       else cycles_means_dir
    return load_cycles_data(dataset, cycles_datas_dir=cycles_means_dir)

def save_cycles_data(cycles_data, cycles_datas_dir=None):
    filepath = get_cycles_data_filepath(cycles_data,
                                        cycles_datas_dir=cycles_datas_dir,
                                        makedirs=True)
    with open(filepath, 'wb') as f:
        pickle.dump(cycles_data, f)

def save_cycles_mean(cycles_mean, cycles_means_dir=None):
    filepath = get_cycles_mean_filepath(cycles_mean,
                                        cycles_means_dir=cycles_means_dir,
                                        makedirs=True)
    with open(filepath, 'wb') as f:
        pickle.dump(cycles_mean, f)

def get_cycles_data(dataset, i=None, results_region_name=None,
                    deactivate_baseline_mod=False, baseline_decimate=1,
                    resolution_sf=None, simulation_settings_file=None,
                    simulations_dir=None, individual_posZ=True, fbnl=True,
                    angles=True, angles_after_processing=True,
                    phi_shift_twopi=True, weighted_energies=False,
                    energy_keys=None, cycles_datas_dir=None, load=False,
                    save=False, **kwargs):
    """
    Load data of all cycles for a dataset
    """
    if load:
        return load_cycles_data(dataset, cycles_datas_dir=cycles_datas_dir)

    # Get stress/release data
    results_region_name = RESULTS_REGION_NAME if results_region_name is None \
                          else results_region_name
    resolution_sf = RESOLUTION_SF if resolution_sf is None else resolution_sf

    # Open experiment
    dbfile = get_dbfile_path(dataset, DATA_BASEDIR)
    exp = pyoti.open_experiment(filename=dbfile, return_last_created=False)
    try:
        mod = exp.modification('baseline_mod')
        if deactivate_baseline_mod:
            mod.active = False
        else:
            mod.iattributes.baseline_decimate = baseline_decimate
    except:
        pass

    with suppress_stdout():
        exp.set_cached_region(results_region_name)

    # Create tether
    results_region = exp.region(results_region_name)
    tether = pyoti.create_tether(region=results_region,
                                 resolution_sf=resolution_sf)
    tether.update()
    # tether.info(i=i)
    number_of_pairs = len(tether.stress_release_pairs()['stress']['idx'])
    dataset['number_of_pairs'] = number_of_pairs

    # Get pair data and simulation for all cycles
    if i is None:
        i = range(number_of_pairs)
    if not isinstance(i, Iterable):
        i = [i]
    I = i

    # Define closure for getting ith cycle
    def _cycle_data(i):
        return _get_cycle_data(dataset, tether, i,
                             simulation_settings_file=simulation_settings_file,
                             simulations_dir=simulations_dir,
                             individual_posZ=individual_posZ,
                             fbnl=fbnl, angles=angles,
                             angles_after_processing=angles_after_processing,
                             phi_shift_twopi=phi_shift_twopi,
                             weighted_energies=weighted_energies,
                             energy_keys=energy_keys,
                             **kwargs)
    try:
        cycles_data = {
            'key': dataset['key'],
            'settings': {
                'dataset': dataset,
                'results_region_name': results_region_name,
                'deactivate_baseline_mod': deactivate_baseline_mod,
                'baseline_decimate': baseline_decimate,
                'resolution_sf': resolution_sf,
                'simulation_settings_file': simulation_settings_file,
                'simulations_dir': simulations_dir,
                'individual_posZ': individual_posZ,
                'fbnl': fbnl,
                'angles': angles,
                'angles_after_processing': angles_after_processing,
                'phi_shift_twopi': phi_shift_twopi,
                'weighted_energies': weighted_energies,
                'energy_keys': energy_keys,
                'kwargs': kwargs
            }
        }
        _cycles_data = [_cycle_data(i) for i in I]
        cycles_data.update(dict(zip(I, _cycles_data)))
    finally:
        # Abort changes and close experiment
        exp.close(abort=True)

    if save:
        save_cycles_data(cycles_data,
                         cycles_datas_dir=cycles_datas_dir)

    return cycles_data


def _get_cycle_data(dataset, tether, i, simulation_settings_file=None,
                    simulations_dir=None, individual_posZ=True, fbnl=True,
                    angles=True, angles_after_processing=True,
                    phi_shift_twopi=True, weighted_energies=False,
                    energy_keys=None, **kwargs):
    """
    Parameters
    ----------
    energy_keys : list of str
        possible energy keys:
            depending on extension: 'e_ext_ssDNA', 'e_ext_dsDNA', 'e_unzip_DNA',
            depening on displacement: 'e_lev'
    """
    # Short notation for tether
    t = tether

    # Get force/extension stress/release force extension data and angles, and
    # fbnl_filter or bin the data considerung the correction factor for kappa_z
    kappa_z_factor = dataset['kappa_z_factor']
    fXYZ_factors = np.array([1, 1, kappa_z_factor])
    if fbnl:
        processing_function = fbnl_force_extension
        process_kwargs = {
            'filter_length_e': 5e-9,  # m
            'edginess': 1,  # int
            'angles_after_filter': angles_after_processing }
        # data, data_filtered, fbnl_filters
        result_data_key = 'data_filtered'
    else:
        processing_function = binned_force_extension
        process_kwargs = {
            'bin_width_e': 5e-9,  # m
            'sortcolumn': 0,  # 0: time, 1: extension
            'angles_after_binning': angles_after_processing }
        # edges, centers, width, bin_Ns, bin_means, bin_stds
        result_data_key = 'bin_means'
    process_kwargs.update(kwargs)
    msg = 'Process data for cycle i = {} ...                  \r'.format(i)
    print(msg, end='', flush=True)
    result = processing_function(t, i, fXYZ_factors=fXYZ_factors,
                                 angles=angles,
                                 phi_shift_twopi=phi_shift_twopi,
                                 **process_kwargs)

    # Combine filtered data, raw data, info (pair) from tether object and
    # excited axis
    pair = t.stress_release_pairs(i=i)
    axis = {'x': 0, 'y': 1}
    ax = pair['stress']['info'][0,0]
    excited_axis = axis[ax]
    data = {}
    data['settings'] = result['settings']
    data['excited_axis'] = excited_axis
    data.update(pair)
    for cycle in ['stress', 'release']:
        for key, value in result['data'][cycle].items():
            data[cycle][key + '_raw'] = value
        data[cycle].update(result[result_data_key][cycle])

    # Get/do the simulation considering the correction factor for kappa_z
    simulation_settings_file = SIMULATION_SETTINGS_FILE \
        if simulation_settings_file is None else simulation_settings_file
    simulations_dir = SIMULATIONS_DIR if simulations_dir is None \
        else simulations_dir
    kappa_z_factor = dataset['kappa_z_factor']
    if individual_posZ:
        posZ = None
    else:
        posZ = np.median(t.get_data('positionZ', samples=None))
    msg = 'Get simulation for cycle i = {} ...                \r'.format(i)
    print(msg, end='', flush=True)
    simulation = get_simulation(t, i, simulation_settings_file, posZ=posZ,
                                individual_posZ=individual_posZ,
                                kappa_z_factor=kappa_z_factor,
                                excited_axis=excited_axis,
                                simulations_dir=simulations_dir)
    sim_key = uzsi.get_key(**simulation['settings'])
    msg = 'Get simulation values for cycle i = {} ...         \r'.format(i)
    print(msg, end='', flush=True)
    sim_values = uzsi.get_simulation_values(simulation, fe_xyz=True,
                                            weighted_energies=weighted_energies,
                                            energy_keys=energy_keys)

    # Calculate normalized energy gains:
    # Calculate energy and extension gains for every point of the simulation
    # and normalize energy by extension gains. As the points of energy gain lay
    # between the simulated extension points, interpolate energy gains by
    # weighting each energy gain difference with its corresponding extensions.
    if weighted_energies:
        msg = 'Calculate normalized energies for cycle i = {} ... \r'.format(i)
        print(msg, end='', flush=True)
        ex_diff = np.diff(sim_values['extension'])
        sim_keys = sim_values.keys()
        e_keys = [sim_key for sim_key in sim_keys if sim_key.startswith('e_')]
        for ek in e_keys:
            # Calculate energy gain from one point of extension to the next
            e_diff = np.diff(sim_values[ek])
            # Normalize energy gain by extension gain
            e_per_m = e_diff / ex_diff
            # Calculate difference between energy gains
            e_per_m_diff = np.diff(e_per_m)
            # Calculate weight for interpolation between steps
            weight = ex_diff[:-1] / (ex_diff[:-1] + ex_diff[1:])
            # Calculate interpolated energy gain difference between the points
            e_per_m_diff_intp = e_per_m_diff * weight
            # Calculate interpolated energy gain
            e_per_m_intp = e_per_m[:-1] + e_per_m_diff_intp
            # Pad unknown ends of energy gain
            e_per_m_intp = np.r_[e_per_m_intp[0],
                                 e_per_m_intp,
                                 e_per_m_intp[-1]]
            sim_values['{}_per_m'.format(ek)] = e_per_m_intp

    data['simulation'] = { 'key': sim_key }
    data['simulation']['settings'] = simulation['settings']
    data['simulation'].update(sim_values)

    return data


#def add_idcs(cycle_data, cycle=None, **kwargs):
#    if cycle is None:
#        cycles = ['stress', 'release', 'simulation']
#    else:
#        cycles = [cycle]
#    for cycle in cycles:
#        cycle_data[cycle]['idcs'] = get_idcs(cycle_data, cycle=cycle, **kwargs)
#    return cycle_data


def get_idcs(cycle_data, cycle='stress', min_x=None, max_x=None,
             include_bounds=True, threshold_f=None, max_length_x=None):
    x = cycle_data[cycle]['extension']
    f = cycle_data[cycle]['force']

    idx_sort = x.argsort()
    xs = x[idx_sort]
    fs = f[idx_sort]

    idx_x = min_max_idx(
        xs, min_x=min_x, max_x=max_x, include_bounds=include_bounds)
    #idx_f = min_max_idx(
    #    f[idx_sort], min_x=min_f, max_x=max_f, include_bounds=include_bounds)
    # TODO: Include bool parameter `continuous`, that lets the `step_idx()`
    #       function only return the index of a step that is followed by a
    #       continuous plateau until the end of the signal and not just the
    #       index of the force exceeding the threshold the first time.
    idx_f = step_idx(fs, threshold_f, include_bounds=include_bounds)
    idx_crop = np.logical_and(idx_x, idx_f)

    try:
        first_x = xs[idx_crop][0]
    except IndexError:
        first_x = 0
    idx_length_x = compare_idx(xs - first_x, max_length_x, comparison='less')
    idx_crop = np.logical_and(idx_crop, idx_length_x)

    idx_sort_crop = idx_sort[idx_crop]
    idx_crop = idx_sort_crop.copy()
    idx_crop.sort()

    return_value = {
        'settings': {
            'cycle': cycle,
            'min_x': min_x,
            'max_y': max_x,
            'include_bounds': include_bounds,
            'threshold_f': threshold_f,
            'max_length_x': max_length_x
        },
        'crop': idx_crop,
        'xsort': idx_sort,
        'xsort_crop': idx_sort_crop,
        'valid_x': idx_x,
        'valid_f': idx_f
    }

    return return_value


# define functions for shift_x determination and mean calculation
def correlate_template(data, template, mode='valid', demean=True,
                       normalize='full', method='auto'):
    """
    Reference:
    https://anomaly.io/understand-auto-cross-correlation-normalized-shift/index.html
    https://github.com/trichter/xcorr

    Normalized cross-correlation of two signals with specified mode.
    If you are interested only in a part of the cross-correlation function
    around zero shift use :func:`correlate_maxlag` which allows to
    explicetly specify the maximum lag.
    :param data,template: signals to correlate. Template array must be shorter
        than data array.
    :param normalize:
        One of ``'naive'``, ``'full'`` or ``None``.
        ``'full'`` normalizes every correlation properly,
        whereas ``'naive'`` normalizes by the overall standard deviations.
        ``None`` does not normalize.
    :param mode: correlation mode to use.
        See :func:`scipy.signal.correlate`.
    :param bool demean: Demean data beforehand.
        For ``normalize='full'`` data is demeaned in different windows
        for each correlation value.
    :param method: correlation method to use.
        See :func:`scipy.signal.correlate`.
    :return: cross-correlation function.
    .. note::
        Calling the function with ``demean=True, normalize='full'`` (default)
        returns the zero-normalized cross-correlation function.
        Calling the function with ``demean=False, normalize='full'``
        returns the normalized cross-correlation function.
    """
    data = np.asarray(data)
    template = np.asarray(template)
    lent = len(template)
    if len(data) < lent:
        raise ValueError('Data must not be shorter than template.')
    if demean:
        template = template - np.mean(template)
        if normalize != 'full':
            data = data - np.mean(data)
    cc = scipy.signal.correlate(data, template, mode, method)
    if normalize is not None:
        tnorm = np.sum(template ** 2)
        if normalize == 'naive':
            norm = (tnorm * np.sum(data ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize == 'full':
            pad = len(cc) - len(data) + lent
            if mode == 'same':
                pad1, pad2 = (pad + 2) // 2, (pad - 1) // 2
            else:
                pad1, pad2 = (pad + 1) // 2, pad // 2
            data = _pad_zeros(data, pad1, pad2)
            # in-place equivalent of
            # if demean:
            #     norm = ((_window_sum(data ** 2, lent) -
            #              _window_sum(data, lent) ** 2 / lent) * tnorm) ** 0.5
            # else:
            #      norm = (_window_sum(data ** 2, lent) * tnorm) ** 0.5
            # cc = cc / norm
            if demean:
                norm = _window_sum(data, lent) ** 2
                if norm.dtype == float:
                    norm /= lent
                else:
                    norm = norm / lent
                np.subtract(_window_sum(data ** 2, lent), norm, out=norm)
            else:
                norm = _window_sum(data ** 2, lent)
            norm *= tnorm
            if norm.dtype == float:
                np.sqrt(norm, out=norm)
            else:
                norm = np.sqrt(norm)
            mask = norm <= np.finfo(float).eps
            if cc.dtype == float:
                cc[~mask] /= norm[~mask]
            else:
                cc = cc / norm
            cc[mask] = 0
        else:
            msg = "normalize has to be one of (None, 'naive', 'full')"
            raise ValueError(msg)
    return cc
def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)
def _window_sum(data, window_len):
    """Rolling sum of data"""
    window_sum = np.cumsum(data)
    # in-place equivalent of
    # window_sum = window_sum[window_len:] - window_sum[:-window_len]
    # return window_sum
    np.subtract(window_sum[window_len:], window_sum[:-window_len],
                out=window_sum[:-window_len])
    return window_sum[:-window_len]


def get_cycle_mean_data(cycle_data, cycle, keys, idx=None, shift_x=None,
                        edges=None, resolution=None):
    """
    Parameters
    ----------
    resolution : float
        The resolution in m⁻¹ of the binned data. The edges of the bins are
        determined by using the extension of the `edges_cycle`. Defaults to
        1/5e-9.
    """
    # Concatenate extension, force, etc. into one data array with extension
    # in the first column
    data_array, keys, columns = concatenate_data_dict(cycle_data[cycle], keys)

    # Select subset of data according to `idx_key`
    if idx is not None:
        data_array = data_array[idx]

    # Shift extension to align with optionally provided edges and revert after
    # having binned the data
    shift_x = 0 if shift_x is None else shift_x
    data_array[:,0] += shift_x

    # Calculate bin means, cast values of simulation mpmath.mpf to float,
    resolution = 1/5e-9 if resolution is None else resolution
    result = calculate_bin_means(data_array.astype(float), bins=edges,
                                 resolution=resolution)
    edges = result['edges']
    centers = result['centers']
    bin_means = result['bin_means']

    # Select values that are valid
    try:
        valid = np.logical_and(edges[:-1] >= data_array[:,0].min(),
                               edges[1:] <= data_array[:,0].max())
    except ValueError:
        valid = np.array([], dtype=int)

    # Separata data array into dictionary and revert shift of extension
    data = separate_data_array(bin_means[valid], keys, columns)
    data['extension'] -= shift_x

    # Add values not contained in bin_means
    data['ext_centers'] = centers[valid] - shift_x
    data['shift_x'] = shift_x
    data['edges'] = edges - shift_x
    data['resolution'] = resolution
    data['sim_idx'] = np.where(valid)[0]

    return data


def get_aligned_cycle_mean(cycle_data, min_x=None, max_length_x=None,
                           threshold_f=None, search_window_e=None,
                           resolution_shift_x=None, edges=None,
                           resolution=None)
    # Set default resolution to 1/5e-9
    resolution = 1/5e-9 if resolution is None else resolution
    resolution_shift_x = 1/5e-9 \
        if resolution_shift_x is None else resolution_shift_x

    # Determine shift_x to align 'simulation' with 'stress' and 'release' cycle
    try:
        align_x = _get_shift_x(cycle_data, min_x=min_x,
                               max_length_x=max_length_x,
                               threshold_f=threshold_f,
                               resolution=resolution_shift_x,
                               search_window_e=search_window_e, plot=False)
    except ValueError:
        align_x = 0
    _cycle_keys = ['extension', 'force', 'displacementXYZ', 'forceXYZ',
                   'positionXYZ', 'distanceXYZ', 'angle_extension',
                   'angle_force', 'angle_extension_after', 'angle_force_after']
    _sim_keys = ['extension', 'force', 'displacementXYZ', 'forceXYZ', 'nuz',
                 'e_ext_ssDNA_per_m', 'e_ext_dsDNA_per_m',
                 'e_unzip_DNA_per_m', 'e_lev_per_m']
    cycle_keys = [key for key in _cycle_keys if key in
                  cycle_data['stress'].keys()]
    sim_keys = [key for key in _sim_keys if key in
                cycle_data['simulation'].keys()]
    cycle_mean = {}
    shift_x = 0
    for cycle, keys in zip(['simulation', 'stress', 'release'],
                           [sim_keys, cycle_keys, cycle_keys]):
        cycle_mean[cycle] = get_cycle_mean_data(cycle_data, cycle, keys,
                                                shift_x=shift_x, edges=edges,
                                                resolution=resolution)
        # Align 'stress' and 'release' with simulation, i.e. for the next cycle
        # set shift_x to calculated one and edges to the ones from 'simulation'
        shift_x = align_x
        edges = cycle_mean[cycle]['edges']

    # Add missing keys
    cycle_mean['simulation']['settings'] = cycle_data['simulation']['settings']
    cycle_mean['stress']['info'] = cycle_data['stress']['info']
    cycle_mean['release']['info'] = cycle_data['release']['info']

    return cycle_mean


def _get_shift_x(cycle_data, min_x=None, max_length_x=None, threshold_f=None,
                 resolution=None, search_window_e=None, peak_height=0.95,
                 plot=False):
    """
    Determine the shift between the 'stress' and the 'simulation' cycle,
    assuming the simulation having the true extension.

    Parameters
    ----------
    cycle_data : dict
    resolution : float
        The resolution in m⁻¹ the cycle_data gets binned with before cross
        correlation.
    search_window_e : float or (float, float)
        Extension in m the stress signal should be shifted up to the left and
        right from the position where the force of the stress and simulation
        are the same. The range is used to calculate the normalized
        cross-correlation and find the position with the best correlation. If
        search_window is None, the whole simulation is searched for the best
        correlation.
    min_x : float
    max_length_x : float
    threshold_f : float
    """
    # Get indices to crop 'stress' force extension curve, assuming the
    # unzipping starts directly after having reached `threshold_f` and is
    # `max_length_x` long. The region of the cropped 'stress' cycle needs to be
    # fully included in the 'simulaton' cycle, otherwise the correlation of
    # 'stress' and 'simulation' fails!
    crop_idx = get_idcs(cycle_data, cycle='stress', min_x=min_x, max_x=None,
                        threshold_f=threshold_f,
                        max_length_x=max_length_x)['crop']

    # Get binned force extension values of simulation and stress
    # Bin the data acccording to the 'simulation' and take the calculated edges
    # from the simulation also for the binning of 'stress'
    # Use only the 'cropped' region of stress and release, as these should
    # contain the unzipping region.
    # No further sorting necessary, as bin means are already sorted along e
    idx = None
    edges = None
    keys = ['extension', 'force']
    cycle_means = {}
    for cycle in ['simulation', 'stress']:
        data = get_cycle_mean_data(cycle_data, cycle, keys, idx=idx,
                                   edges=edges, resolution=resolution)

        cycle_means[cycle] = data

        # Set edges for the 'stress' cycle to the ones from 'simulation' and
        # idx to the calculated one according to the calculated cropping
        idx = crop_idx
        edges = data['edges']

    if len(cycle_means['stress']['ext_centers']) == 0:
        msg1 = 'No datapoints of stress cycle where selected!'
        msg2 = 'Provide proper `min_x`, `max_length_x`, and `threshold_f`!'
        raise ValueError(msg1, msg2)
    if len(cycle_means['stress']['ext_centers']) \
           >= len(cycle_means['simulation']['ext_centers']):
        msg1 = 'Length of simulation needs to be greater then cropped stress cycle!'
        msg2 = 'Provdie proper `min_x`, `max_length_x`, and `threshold_f`'
        msg3 = 'Or provide simulation with more datapoints!'
        raise ValueError(msg1, msg2, msg3)

    if search_window_e is not None:
        # Find index of simulation, where force is same as first force of stress
        start_stress_f = cycle_means['stress']['force'][0]
        start_sim_idx = np.argmax(cycle_means['simulation']['force'] \
                                  >= start_stress_f)
        start_sim_e = cycle_means['simulation']['extension'][start_sim_idx]

        # Get indices of simulation where extension is less than first and
        # greater than length of stress according to search_window_e
        try:
            search_window_left_e = search_window_e[0]
            search_window_right_e = search_window_e[1]
        except:
            search_window_left_e = search_window_e
            search_window_right_e = search_window_e
        min_sim_e = start_sim_e - search_window_left_e
        max_sim_e = start_sim_e + search_window_right_e \
                    + (cycle_means['stress']['extension'][-1] \
                    - cycle_means['stress']['extension'][0])
        min_sim_idx = cycle_means['simulation']['extension'] >= min_sim_e
        max_sim_idx = cycle_means['simulation']['extension'] <= max_sim_e
        sim_idx = np.logical_and(min_sim_idx, max_sim_idx)
    else:
        sim_idx = slice(None)

    # Correlate forces of 'stress' and 'simulation'
    a = cycle_means['simulation']['force'][sim_idx]
    b = cycle_means['stress']['force']
    #corr = np.correlate(a, b, 'valid')
    #corr = np.convolve(a, b, 'valid')
    #corr = match_template(np.atleast_2d(a)/a.max(), np.atleast_2d(b)/a.max()).T[:,0]
    # divide by maximum value to prevent errors due to small float values
    corr = correlate_template(a/a.max(), b/b.max())

    # Find relative shift of simulation to stress in datapoints
    max_peak_idx = np.argmax(corr)
    # Try to find peaks with a minimum amplitude of 0.95 x the maximum
    # correlation
    peaks = scipy.signal.find_peaks(corr, corr[max_peak_idx]*peak_height)[0]
    # If there are now peaks comparable to the height of the maximum
    # correlation, choose the shift_x to be according to the position with same
    # force of simulation and stress. Otherwise
    if len(peaks) == 0 and search_window_e is not None:
        shift_x_idx = start_sim_idx - np.argmax(min_sim_idx)
    else:
        shift_x_idx = max_peak_idx

    # Get shift of extension in m
    shift_x = cycle_means['simulation']['extension'][sim_idx][shift_x_idx] \
                                        - cycle_means['stress']['extension'][0]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(corr)
        ax.plot(shift_x_idx, corr[shift_x_idx], 'o')
        ax.plot(peaks, corr[peaks], '.')
        #ax2 = ax.twinx()
        #ax2.plot(corr2)
        #ax2.plot(np.diff(corr), '.')
        fig.show()

    return shift_x


def add_areas(cycle_data):
    """
    Parameters
    ----------
    cycle : str
        'stress', 'release', or 'simulation'. Defaults to ['stress, 'release',
        'simulation'].
    """
    cycles = ['stress', 'release', 'simulation']
    for cycle in cycles:
        cycle_data[cycle]['area'] = get_area(cycle_data, cycle=cycle,
                                             x_key='extension',
                                             integration_type='trapz')
        cycle_data[cycle]['cumarea'] = get_area(cycle_data, cycle=cycle,
                                                x_key='extension',
                                                integration_type='cumtrapz')
        try:
            cycle_data[cycle]['rectarea'] = get_area(cycle_data, cycle=cycle,
                                                     x_key='ext_centers',
                                                     integration_type='rect')
        except KeyError as e:
            msg = "x_key {} does not exist in `cycle_data`".format(e)
            warnings.warn(msg)
    return cycle_data


def get_area(cycle_data, cycle='stress', x_key=None, idx=None,
             integration_type=None, resolution=None):
    """
    Parameters
    ----------
    x_key : str
        Key to choose the values for x axis. Defaults to 'extension'.
    idx : indices
        Indices to use for area calculation.
    resolution : float
        Resolution is only used to calculate 'rectarea' and if `cycle_data`
        does not provide a resolution. Defaults to 1.
    """
    idx = slice(None) if idx is None else idx
    x_key = 'extension' if x_key is None else x_key
    x = cycle_data[cycle][x_key][idx]
    y = cycle_data[cycle]['force'][idx]

    if 'resolution' in cycle_data[cycle]:
        resolution = cycle_data[cycle]['resolution']
    else:
        resolution = 1 if resolution is None else resolution
    integration_fs = {
        'simps': simps,
        'trapz': np.trapz,
        'cumtrapz': cumtrapz,
        'rect': lambda y, x: y / resolution
    }
    initial = 0
    integration_kwargs = {
        'cumtrapz': { 'initial': initial }
    }
    f = integration_fs.get(integration_type, np.trapz)
    if integration_type not in integration_fs:
        integration_type = 'trapz'
    f_kwargs = integration_kwargs.get(integration_type, {})

    try:
        if f_kwargs:
            area = f(y, x, **f_kwargs)
        else:
            area = f(y, x)
    except (IndexError, ValueError):
        integration_default = {
            'simps': 0.0,
            'trapz': 0.0,
            'cumptrapz': np.array([]),
            'rect': np.array([])
        }
        area = integration_default.get(integration_type, 0.0)


    return_value = {
        'value': area,
        'type': integration_type,
        'idx': idx
    }

    return return_value


def plot_unzipping(x, f, x_sim=None, f_sim=None, nuz=None, x_release=None,
                   f_release=None, ax=None, xlim=None, ylim=None, xlabel=True,
                   xticklabel=True, ylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot simulated and measured unzipping curve
    if x_sim is not None and f_sim is not None:
        ax.plot(x_sim * 1e9, f_sim * 1e12, linestyle=':',
                label='Force microsphere')
        ax.set_prop_cycle(None)  # reset color cycler
    ax.plot(x * 1e9, f * 1e12)
    if x_release is not None and f_release is not None:
        ax.plot(x_release * 1e9, f_release * 1e12)
    if xlabel:
        ax.set_xlabel('(Apparent) ext of construct (nm)')
    if ylabel:
        ax.set_ylabel('Force (pN)')

    # Plot number of simulated unzipped basepairs
    if x_sim is not None and nuz is not None:
        ax2 = ax.twinx()
        ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        ax2.plot(x_sim * 1e9, nuz, color='cyan')
        if ylabel:
            ax2.set_ylabel('# unzip bps')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, (ax, ax2)


def plot_unzipping_3D(x, fXYZ, x_sim=None, fXYZ_sim=None, excited_axis=0,
                      x_release=None, fXYZ_release=None, ax=None, xlim=None,
                      ylim=None, xlabel=True, xticklabel=True, ylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot individual simulated and measured unzipping forces
    if x_sim is not None and fXYZ_sim is not None:
        if excited_axis == 1:
            ax.plot([])
        ax.plot(x_sim * 1e9, fXYZ_sim[:,0] * 1e12, linestyle=':')
        if excited_axis == 0:
            ax.plot([])
        ax.plot(x_sim * 1e9, fXYZ_sim[:,1] * 1e12, linestyle=':')
        ax.set_prop_cycle(None)  # reset color cycler
    ax.plot(x * 1e9, np.abs(fXYZ) * 1e12)
    if x_release is not None and fXYZ_release is not None:
        ax.plot(x_release * 1e9, np.abs(fXYZ_release * 1e12))
    if xlabel:
        ax.set_xlabel('(Apparent) ext of construct (nm)')
    if ylabel:
        ax.set_ylabel('Force (pN)')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax


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


def plot_angles_fe(x, ext_theta_phi, force_theta_phi, ax=None, xlim=None,
                   xlabel=True, xticklabel=True, legend=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    lns1 = ax.plot(x * 1e9, ext_theta_phi[:, 0], label=r'$\theta$ E')
    lns2 = ax.plot(x * 1e9, force_theta_phi[:, 0], label=r'$\theta$ F')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%3d'))
    if xlabel:
        ax.set_xlabel('Apparent extension (nm)')
    ax.set_ylabel(r'$\Theta (°)$')

    # ax2 = _create_twin_ax(ax)
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    lns3 = ax2.plot(x * 1e9, ext_theta_phi[:, 1], label=r'$\phi$ E')
    lns4 = ax2.plot(x * 1e9, force_theta_phi[:, 1], label=r'$\phi$ F')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%3d'))
    ax2.set_ylabel(r'$\Phi (°)$')

    ax.tick_params(labelbottom=xticklabel)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if legend:
        lns = list(itertools.chain(lns1, lns2, lns3, lns4))
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

    return fig, (ax, ax2)


def plot_unspec_bounds(cycle_data, bps_A=None, bps_B=None, axes=None):
    if axes is None:
        fig, ax = plt.subplots()
        axes = [ax]
    else:
        fig = axes[0].get_figure()

    # Plot the extension up to where the not yet unzipped BPs could
    # unspecifically bind
    E = cycle_data['simulation']['extension'].astype(float)
    F = cycle_data['simulation']['force'].astype(float)
    NUZ = cycle_data['simulation']['nuz'].astype(float)
    s = cycle_data['simulation']['settings']
    S = s['S']
    L_p_ssDNA = s['L_p_ssDNA']
    z = s['z']
    L_p_dsDNA = s['L_p_dsDNA']
    pitch = s['pitch']
    T = s['T']
    h0 = s['h0']
    radius = s['radius']
    bps = len(s['bases'])

    # Plot the extension up to where the not yet unzipped BPs could
    # unspecifically bind to the microsphere. The number of unzipped basepairs
    # up to where not yet unzipped BPs could bind is determined assuming a
    # force corresponding to the unzipping force
    if bps_A is not None:
        A = []
        E_ssDNA = []
        D_fork = []
        E_unzip = []
        for nuz, f in zip(NUZ, F):
            a = uzsi.ext_dsDNA_wlc(f, bps_A, pitch, L_p_dsDNA, T)
            e_ssDNA = uzsi.ext_ssDNA(f, nuz, S, L_p_ssDNA, z, T=T)
            A.append(a)
            E_ssDNA.append(e_ssDNA)
            D_fork.append(a + e_ssDNA)
            E_unzip.append(uzsi.ext_dsDNA_wlc(f, bps - nuz, pitch, L_p_dsDNA, T))
        A = np.array(A)
        E_ssDNA = np.array(E_ssDNA)
        D_fork = np.array(D_fork)
        E_unzip = np.array(E_unzip)
        D_tip = D_fork - E_unzip
        try: idx = np.where(D_tip <= 0)[0][-1]
        except: idx = 0
        e_tip = E[idx]
        for ax in axes:
            ax.axvline(x=e_tip*1e9, linestyle=':', color='black',
                       linewidth=0.5)

    # Plot the extension up to where the tip of the unzipped BPs could
    # unspecifically bind to the glass surface. Assumption: the force acting on
    # the unzipped BPs and causing the extension is set to the unzipping force,
    # which is an upper estimate of the maximum to expected force acting on the
    # BPs unspecifically bound to the glass suface. Due to the geometry, the
    # force should not exceed the unzipping force.
    if bps_A is not None and bps_B is not None:
        B = []
        for nuz, f in zip(NUZ, F):
            #A.append(uzsi.ext_dsDNA_wlc(f, bps_A, pitch, L_p_dsDNA, T))
            B.append(uzsi.ext_dsDNA_wlc(f, bps_B, pitch, L_p_dsDNA, T))
            #E_ssDNA.append(uzsi.ext_ssDNA(f, nuz, S, L_p_ssDNA, z, T=T))
            #E_unzip.append(uzsi.ext_dsDNA_wlc(f, bps - nuz, pitch, L_p_dsDNA, T))
        B = np.array(B)
        C = A + B + 2 * E_ssDNA + radius
        # height H, i.e. distance between glass-surface - bead-surface
        H = h0 - cycle_data['simulation']['displacementXYZ'][:,1].astype(float)
        # distance D, i.e. distance of glass-surface - bead-center
        D = H + radius
        # height H_fork, i.e. distance of glass-surface - unzipping fork
        H_fork = D * (B + E_ssDNA) / C
        # height H_tip, i.e. distance of glass-surface - unzipping tip/hairpin
        # for highest force
        H_tip = H_fork - E_unzip
        try: idx = np.where(H_tip <= 0)[0][-1]
        except: idx = 0
        e_tip = E[idx]
        for ax in axes:
            ax.axvline(x=e_tip*1e9, linestyle='--', color='black',
                       linewidth=0.5)

    return fig, axes


def plot_cycle_data(cycle_data, release=False, bps_A=None, bps_B=None,
                    shift_x=None, print_shift_x=True, xlim=None, ylim=None):
    # Plot measured and simulated force extension in 3D and angles of force and
    # extension
    shift_x_stress = 0
    shift_x_release = 0
    if 'shift_x' in cycle_data['stress']:
        shift_x_stress = cycle_data['stress']['shift_x']
    if release and 'shift_x' in cycle_data['release']:
        shift_x_release = cycle_data['release']['shift_x']
    # shift_x parameter takes precedence over other shift_x settings
    if shift_x is not None:
        shift_x_stress = shift_x
        shift_x_release = shift_x

    # Get the unzipping data
    data = cycle_data
    excited_axis = data['excited_axis']
    x = data['stress']['extension'] + shift_x_stress
    f = data['stress']['force']
    fXYZ = np.abs(data['stress']['forceXYZ'])
    if release:
        x_release = data['release']['extension'] + shift_x_release
        f_release = data['release']['force']
        fXYZ_release = np.abs(data['release']['forceXYZ'])
    else:
        x_release = None
        f_release = None
        fXYZ_release = None
    x_sim = data['simulation']['extension']
    f_sim = data['simulation']['force']
    fXYZ_sim = data['simulation']['forceXYZ']
    nuz = data['simulation']['nuz']
    # Get the angles of force and extension vectors
    # 0: theta_extension, 1: phi_exension
    ext_theta_phi = data['stress']['angle_extension_after']
    # 0: theta_force, 1: phi_force
    force_theta_phi = data['stress']['angle_force_after']

    fig, axes = plt.subplots(3, 1)

    plot_unspec_bounds(cycle_data, bps_A=bps_A, bps_B=bps_B, axes=axes)
    plot_unzipping(x, f, x_sim=x_sim, f_sim=f_sim, nuz=nuz,
                   x_release=x_release, f_release=f_release, ax=axes[0],
                   xlim=xlim, ylim=ylim, xlabel=False, xticklabel=False,
                   ylabel=True)
    plot_unzipping_3D(x, fXYZ, x_sim=x_sim, fXYZ_sim=fXYZ_sim,
                      excited_axis=excited_axis, x_release=x_release,
                      fXYZ_release=fXYZ_release, ax=axes[1], xlim=xlim,
                      ylim=ylim, xlabel=False, xticklabel=False, ylabel=True)
    plot_angles_fe(x, ext_theta_phi, force_theta_phi, ax=axes[2], xlim=xlim)

    if print_shift_x:
        ax = axes[0]
        shift_x = shift_x_stress
        ax.text(0.98, 0.03, r'{:.0f}$\,$nm'.format(shift_x*1e9), fontsize=7,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)

    # Link y-axes of force plots together
    axes[0].get_shared_y_axes().join(axes[0], axes[1])
    for ax in axes[:-1]:
        # link x-axes together
        ax.get_shared_x_axes().join(ax, axes[-1])
        ax.tick_params(bottom=False)
    for ax in axes[1:]:
        ax.tick_params(top=False)

    axes[0].set_title('Unzipping force extensions curves and angles')

    return fig, axes
