import itertools
import numpy as np
import os
import operator as op
import pyoti
import unzipping_simulation as uzsi
import warnings

from collections.abc import Iterable
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import simps, cumtrapz

from .helpers import compare_idx, min_max_idx, step_idx
from .force_extension import binned_force_extension, fbnl_force_extension, \
                             get_simulation
from .helpers import suppress_stdout

DATA_BASEDIR = './data'
SIMULATION_SETTINGS_FILE = './standard_simulation_settings.p'
SIMULATIONS_DIR = './simulations'
RESULTS_REGION_NAME = 'results'
RESOLUTION_SF = 1000


def create_dataset(directory, dbname, kappa_z_factor, shift_x, c_rad52=0,
                   c_count=0, number_of_pairs=0, key=None, datadir='data'):
    directory = os.path.join(directory, datadir)
    dataset = {
        'directory': directory,
        'dbname': dbname,
        'kappa_z_factor': kappa_z_factor,
        'shift_x': shift_x,
        'c_rad52': c_rad52,
        'c_count': c_count,
        'number_of_pairs': number_of_pairs,
        'key': key,
    }
    return dataset


def get_dbfile_path(dataset, basedir=None):
    basedir = '.' if basedir is None else basedir
    directory = dataset['directory']
    dbname = dataset['dbname']
    fullpath = os.path.join(basedir, directory, dbname + '.fs')
    return fullpath


def get_cycles_data(dataset, i=None, results_region_name=None,
                    deactivate_baseline_mod=False, baseline_decimate=1,
                    resolution_sf=None, simulation_settings_file=None,
                    simulations_dir=None, individual_posZ=True, fbnl=True,
                    angles=True, extra_traces=None,
                    angles_after_processing=True, phi_shift_twopi=True,
                    weighted_energies=False, energy_keys=None,
                    **kwargs):
    """
    Load data of all cycles for a dataset
    """
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
    if 'number_of_pairs' in dataset:
        number_of_pairs = dataset['number_of_pairs']
    else:
        number_of_pairs = len(tether.stress_release_pairs()[0])
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
                             extra_traces=extra_traces,
                             angles_after_processing=angles_after_processing,
                             phi_shift_twopi=phi_shift_twopi,
                             weighted_energies=weighted_energies,
                             energy_keys=energy_keys,
                             **kwargs)
    try:
        _cycles_data = [_cycle_data(i) for i in I]
        cycles_data = dict(zip(I, _cycles_data))
    finally:
        # Abort changes and close experiment
        exp.close(abort=True)

    return cycles_data


def _get_cycle_data(dataset, tether, i, simulation_settings_file=None,
                    simulations_dir=None, individual_posZ=True, fbnl=True,
                    angles=True, extra_traces=None,
                    angles_after_processing=True, phi_shift_twopi=True,
                    weighted_energies=False, energy_keys=None,
                    **kwargs):
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
        # data_filtered, fbnl_filters = result
        result_data_idx = 0
    else:
        processing_function = binned_force_extension
        process_kwargs = {
            'bin_width_e': 5e-9,  # m
            'sortcolumn': 0,  # 0: time, 1: extension
            'angles_after_binning': angles_after_processing }
        # bin_edges, bin_centers, bin_widths, bin_means, bin_stds, bin_Ns = result
        result_data_idx = 3
    process_kwargs.update(kwargs)
    result = processing_function(t, i, fXYZ_factors=fXYZ_factors,
                                 angles=angles, extra_traces=extra_traces,
                                 phi_shift_twopi=phi_shift_twopi,
                                 **process_kwargs)
    result_data = result[result_data_idx]

    # Select corresponding bin_means and assign to easy to remember variable
    # names
    data = {}
    # Get excited axis
    axis = {'x': 0, 'y': 1}
    ax = t.stress_release_pairs(i=i, info=True)[2][0,0]
    excited_axis = axis[ax]
    data['excited_axis'] = excited_axis
    for c, cycle_name in zip([0, 1], ['stress', 'release']):
        data[cycle_name] = {}
        d = data[cycle_name]

        idx = 0
        d['time'] = result_data[c][:,idx]; idx += 1
        d['extension'] = result_data[c][:,idx]; idx += 1
        d['force'] = result_data[c][:,idx]; idx += 1
        if angles:
            d['angles_extension'] = result_data[c][:,idx:idx+2]; idx += 2
            d['angles_force'] = result_data[c][:,idx:idx+2]; idx += 2
            d['distanceXYZ'] = result_data[c][:,idx:idx+3]; idx += 3
            d['forceXYZ'] = result_data[c][:,idx:idx+3]; idx += 3
        if extra_traces is not None:
            columns = 3  # time, extension, force
            if angles: columns += 10
            if angles and angles_after_processing: columns += 4
            columns_total = result_data[c].shape[1]
            columns_extra = columns_total - columns
            d['extra'] = result_data[c][:,idx:idx+columns_extra]; idx += columns_extra
        if angles and angles_after_processing:
            d['angles_extension_after'] = result_data[c][:,idx:idx+2]; idx += 2
            d['angles_force_after'] = result_data[c][:,idx:idx+2]; idx += 2

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
    msg = 'Get simulation for cycle i = {} ...\r'.format(i)
    print(msg, end='', flush=True)
    simulation = get_simulation(t, i, simulation_settings_file, posZ=posZ,
                                individual_posZ=individual_posZ,
                                kappa_z_factor=kappa_z_factor,
                                excited_axis=excited_axis,
                                simulations_dir=simulations_dir)
    sim_key = uzsi.get_key(**simulation['settings'])
    #e_keys = [
    #    'e_ext_ssDNA',
    #    'e_ext_dsDNA',
    #    'e_unzip_DNA',
    #    #'e_lev'  # does not depend on extension but rather on distance
    #]
    #energy_keys = e_keys if energy_keys is None else energy_keys
    sim_values = uzsi.get_simulation_values(simulation, fe_xyz=True,
                                            weighted_energies=weighted_energies,
                                            energy_keys=energy_keys)

    # Calculate normalized energy gains:
    # Calculate energy and extension gains for every point of the simulation
    # and normalize energy by extension gains. As the points of energy gain lay
    # between the simulated extension points, interpolate energy gains by
    # weighting each energy gain difference with its corresponding extensions.
    if weighted_energies:
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
    data['simulation'].update(sim_values)
    data['simulation']['settings'] = simulation['settings']

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
    idx_f = step_idx(fs, threshold_f, include_bounds=include_bounds)
    idx_crop = np.logical_and(idx_x, idx_f)

    first_x = xs[idx_crop][0]
    idx_length_x = compare_idx(xs - first_x, max_length_x, comparison='less')
    idx_crop = np.logical_and(idx_crop, idx_length_x)

    idx_sort_crop = idx_sort[idx_crop]
    idx_crop = idx_sort_crop.copy()
    idx_crop.sort()

    return_value = {
        'crop': idx_crop,
        'xsort': idx_sort,
        'xsort_crop': idx_sort_crop,
        'valid_x': idx_x,
        'valid_f': idx_f,
        'settings': {
            'cycle': cycle,
            'min_x': min_x,
            'max_y': max_x,
            'include_bounds': include_bounds,
            'threshold_f': threshold_f,
            'max_length_x': max_length_x
        }
    }

    return return_value


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

    if f_kwargs:
        area = f(y, x, **f_kwargs)
    else:
        area = f(y, x)

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
        H = h0 - cycle_data['simulation']['displacement'][:,1].astype(float)
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
    ext_theta_phi = data['stress']['angles_extension_after']
    # 0: theta_force, 1: phi_force
    force_theta_phi = data['stress']['angles_force_after']

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
