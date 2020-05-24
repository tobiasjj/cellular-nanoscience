#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# dna, functions to describe and fit DNA force extension curves
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
from collections import namedtuple
from lmfit import minimize, Parameters, fit_report
from scipy import constants


k_B = constants.value('Boltzmann constant')
ForceExtension = namedtuple('ForceExtension', ['extension', 'force'])


def worm_like_chain(x, L_0, L_p=50e-9, T=298):
    """
    A worm-like chain model.

    Marko, J.F.; Eric D. Siggia. "Stretching DNA". Macromolecules. 1995. 28:
    8759–8770. doi:10.1021/ma00130a008

    Parameters
    ----------
    x : float
        Extension (m)
    L_0 : float
        Contour length (m)
    L_p : float
        Persistence length (m)
    T : float
        Temperature (K)

    Returns
    -------
    1D numpy.ndarray of type float
        Force (N).
    """
    F = k_B * T / L_p * (1 / (4 * (1 - x / L_0)**2) - 1/4 + x / L_0)

    return F


def worm_like_chain_1p(x, L_0, L_p=50e-9, T=298):
    """
    An improved worm-like chain model.

    Petrosyan, R. "Improved approximations for some polymer extension
    models". Rehol Acta. 2016. doi:10.1007/s00397-016-0977-9

    Parameters
    ----------
    x : float
        Extension (m)
    L_0 : float
        Contour length (m)
    L_p : float
        Persistence length (m)
    T : float
        Temperature (K)

    Returns
    -------
    1D numpy.ndarray of type float
        Force (N).
    """
    F = k_B * T / L_p * (1 / (4 * (1 - x / L_0)**2) - 1/4 + x / L_0
                         - 0.8 * (x / L_0)**2.15)

    return F


# TODO: Check units
def twistable_wlc(F, L_0, x=None, L_p=43.3e-9, K_0=1246e-12, S=1500e-12,
                  C=440e-30, T=298.2):
    """
    The twistable worm-like chain model.

    Gross, P.; Laurens, N.; Oddershede, L. B.; Bockelmann, U.; Peterman, E. J.
    & Wuite, G. J. "Quantifying how DNA stretches, melts and changes twist
    under tension". Nature Physics, Nature Research, 2011, 7, 731-736

    Paramaters
    ----------
    F : float
        force (N)
    L_0 : float
        contour length (m)
    x : float
        extension (m)
    L_p : float
        persistence length (m)
    K_0 : float
        elastic modulus (N)
    S : float
        stretch modulus (N)
    C : float
        twist rigidity (Nm²)
    """
    def g(F, fitting=False, use_lambda_fit=False, use_unwinding_fit=False):
        """
        g(F) describes the twist-stretch coupling how DNA complies to tension.

        The original equation is as follows:
        g(F) = (S * C - C * F * (x/L_0 - 1 + 1/2
                * (k*T/(F*L_p))**(1/2))**(-1))**(1/2)

        The equation can be simplified:
            - below Fc of 30 pN: as a constant (- 100 pN nm)
            - above Fc to linear order: g0 + g1 * F

        g0 was determied to be 590 pN nm (or 560 pN nm with lambda DNA)
        g1 was determined to be 18 nm
        """
        if fitting:
            return (S * C - C * F * (x/L_0 - 1 + 1/2
                    * (k_B*T/(F*L_p))**(1/2))**(-1))**(1/2)
        if F <= 30e-12:  # N
            return - 100e-21  # Nm
        else:
            g0 = - 590e-21  # Nm
            if use_lambda_fit:
                g0 = - 560e-21  # Nm
            if use_unwinding_fit:
                g0 = - 637e-21  # Nm
            return g0 + 17e-9 * F

    x = L_0 * (1 - 1/2 * (k_B * T / (F * L_p))**(1/2)
               + C / (-g(F)**2 + S * C) * F)

    return x


def force_extension(bps, pitch=0.338e-9, L_p=43.3e-9, T=298.2, min_ext=0.5,
                    max_ext=0.978, samples=1000):
    """
    Calculate force extension data for a DNA, assuming a worm-like chain model.

    Parameters
    ----------
    bps : int
        number of base-pairs
    pitch : float
        length of one base-pair (m)
        Defaults to 0.338e-9 m (Saenger 1988)
    L_p : float
        persistence length (m)
    T : float
        temperature (K)
    min_ext : float
        minimum extension, normalized to contour length
        0.5-0.978 entspricht ~520nm-1057nm (0.118...49.2pN)
    max_ext : float
        maximum extension, normalized to contour length
    samples : int
        number of samples to generate

    Returns
    -------
    ForceExtension
        The namedtuple consists of extension and force in (m)
    """
    L_0 = bps * pitch  # m contour length
    x = np.linspace(min_ext * L_0, max_ext * L_0, samples)
    F = worm_like_chain(x, L_0, L_p, T)

    # K_0=1246e-12  # N elastic modulus
    # x = (x + f / K_0) * L_0

    return ForceExtension(extension=x, force=F)


def twistable_force_extension(bps, pitch=0.338e-9, L_p=43.3e-9, K_0=1246e-12,
                              S=1500e-12, C=440e-30, T=298.2, min_f=10e-12,
                              max_f=60e-12, samples=1000):
    F = np.linspace(min_f, max_f, samples)
    L_0 = bps * pitch
    x = twistable_wlc(F, L_0, L_p, K_0, S, C, T)

    return ForceExtension(extension=x, force=F)


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
    # Reduce calculation time, if no min/max values are given
    if min_x is None and max_x is None and min_y is None and max_y is None:
        idx = np.ones_like(x, dtype=bool)
        return idx
    if include_bounds:
        ol = operator.le
        og = operator.ge
    else:
        ol = operator.lt
        og = operator.gt
    max_x = max_x or float('inf')
    min_x = min_x or float('-inf')
    i_x = ol(x, max_x)
    i_x = np.logical_and(i_x, og(x, min_x))
    if y is None:
        return i_x
    max_y = max_y or float('inf')
    min_y = min_y or float('-inf')
    i_y = ol(y, max_y)
    i_y = np.logical_and(i_y, og(y, min_y))
    idx = np.logical_and(i_x, i_y)
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
    idx = crop_x_y_idx(x, y, min_x, max_x, min_y, max_y, include_bounds)
    if y is None:
        return x[idx]
    else:
        return x[idx], y[idx]


def residual(params, model_func, x, data, min_x_param=None, max_x_param=None,
             eps=None):
    """
    Calculate the residuals of a model and given data.

    Parameters
    ----------
    params : lmfit.Parameters
    model_func : function
        Function with the header model_func(x, **params). It should return
        a 1D numpy.ndarray of type float.
    x : 1D numpy.ndarray of type float
    data : 1D numpy.ndarray of type float
    min_x_param : str
        Crop `x` (and `data`) with the function `crop_x_y()` (paramater
        `min_x`) according to the parameter with the key `min_x_param` in
        `params`,  before calculating the model.
    max_x_param : str
        Crop `x` (and `data`) with the function `crop_x_y()` (paramater
        `max_x`) according to the parameter with the key `max_x_param` in
        `params`,  before calculating the model.
    eps : float
    """
    # Crop the X data according to a fit parameter
    if min_x_param is not None or max_x_param is not None:
        min_x = params.get(min_x_param, None)
        max_x = params.get(max_x_param, None)
        x, data = crop_x_y(x, data, min_x=min_x, max_x=max_x,
                           include_bounds=False)

    # Calculate data according to the model function
    model = model_func(x, **params)

    # Calculate the residuals of the model and the given data
    if eps is None:
        return model - data
    return (model - data) / eps


def get_DNA_fit_params(bps, pitch=None, L_p=None, T=None, vary=None, **kwargs):
    """
    Create a default `lmfit.Parameters` instance, which can be used to fit DNA
    force extension data with lmfit.

    The default parameters calculated are the contour length (L_0), the
    persistence length (L_p) and the temperature (T). The contour length is
    calculated from the number of basepairs (bps) and the pitch (pitch) of the
    DNA.

    Parameters
    ----------
    bps : float
        Number of basepairs of the DNA to be fitted. The parameter `bps` is
        used to calculate the initial value (start value) of the contour length
        'L_0' (i.e. `bps`*`pitch`). See also parameter `params`.
    pitch : float (optional)
        Pitch in nm (defaults to 0.338e-9 nm). The parameter `pitch` is
        used to calculate the initial value (start value) of the contour length
        'L_0' (i.e. `bps`*`pitch`). See also parameter `params`.
    L_p : float (optional)
        Persistence length in m (defaults to 0.338e-9 m).
    T : float
        Temperature in K (defaults to 298 K).
    vary : dict (optional)
        A dictionary setting the Parameters, which should be varied during the
        fitting process (defaults to {'L_0': True, 'L_p': True, 'T': False}).
        As an alternative to the key 'L_0', 'bps' can be used.
    **kwargs
        Additional parameters for compatibility reasons that are neglected.

    Returns
    -------
    lmfit.Parameters
        Lmfit Parameters instance containing the Parameter objects. The default
        is a Parameters instance, containing the Parameter objects 'L_0' (i.e.
        `bps`*`pitch`), `L_p`, and `T`, initialized with the corresponding
        function parameters.
    """
    pitch = pitch or 0.338e-9
    L_p = L_p or 50e-9
    T = T or 298
    # Calculate the contour length of the DNA
    L_0 = bps * pitch  # m contour length

    vary = vary or {}
    vary_L_0_default = vary.get('bps', True)

    params = Parameters()
    params.add('L_0', value=L_0, vary=vary.get('L_0', vary_L_0_default))
    params.add('L_p', value=L_p, vary=vary.get('L_p', True))
    params.add('T', value=T, vary=vary.get('T', False))

    return params


def fit_force_extension(e, f, bps, model_func=None, params=None, min_e=None,
                        max_e=None, max_f=None, max_e_dyn_L0=False,
                        verbose=False, return_model_func=False, **kwargs):
    """
    Fit a model function, e.g. a worm-like chain model, to force extension data
    of DNA.

    Parameters
    ----------
    e : 1D numpy.ndarray of type float
        The extension (m).
    f : 1D numpy.ndarray of type float
        The force (N).
    bps : float
        The number of basepairs of the DNA. If you do not know the number of
        basepairs, try an upper estimate to fit the data. The number of
        basepairs is used to calculate the start value for the contour length
        L_0 for the fitting procedure. See also the function
        `get_DNA_fit_params.
    model_func : func
        Set model function, that should have the header model_func(e, **params)
        and return f. Defauts to the function `worm_like_chain`.
    params : lmfit.Parameters
        The parameters to be fitted (defaults to the output of
        `get_DNA_fit_params(bps, **kwargs)`).
    min_e : float
        Minimum extension in m (defaults to float(-inf)) to be used to fit the
        model.
    max_e : float
        Maximum extension in m (defaults to float(+inf)) to be used to fit the
        data. See also parameter `max_e_dyn_L0`.
    max_f : float
        Maximum force in N (defaults to 15e-12) to be used to fit the data.
    max_e_dyn_L0 : bool
        Set `max_e` dynamically to the contour length 'L_0' for every fitting
        evaluation. If L0 is greater than the parameter `max_e`, the value of
        `max_e` will be used, instead.
    verbose : bool
        Be verbose about the fit result.
    return_model_func : bool
        Return the used model_func additionally to the fit result.
    **kwargs
        Arguments passed to the parameter function `get_DNA_fit_params`.

    Returns
    -------
    lmfit.minimizer.MinimizerResult
        If `return_model_func` is False.
    (lmfit.minimizer.MinimizerResult, model_func)
        If `return_model_func` is True
    """
    # Choose the model function and initialize the fit parameters
    model_func = model_func or worm_like_chain
    params = params or get_DNA_fit_params(bps, **kwargs)

    # Crop the data. Choose boundaries to avoid nan values and a max force of
    # 15 pN, up to where the wlc model is valid.
    min_x = min_e
    max_x = max_e
    max_y = max_f or 15e-12
    e, f = crop_x_y(e, f, min_x=min_x, max_x=max_x, max_y=max_y,
                    include_bounds=False)

    # Choose the parameters for the function residual, which calculates the
    # difference of the model function to the given data
    residual_args = model_func, e, f
    residual_kwargs = {}
    if max_e_dyn_L0:
        # Crop e and f data variates to contour length 'L_0', i.e. up to where
        # the wlc model is valid
        residual_kwargs['max_x_param'] = 'L_0'

    # Do the fitting:
    #   minimize -> residual(params, residual_args) -> model_func(e, params)
    out = minimize(residual, params, args=residual_args, kws=residual_kwargs)

    if verbose:
        print(fit_report(out))
        print('[[DNA related info]]')
        print('    Number of base-pairs: {:.0f}'.format(
                            np.round(out.params['L_0'] / params['L_0'] * bps)))

    if return_model_func:
        return out, model_func
    else:
        return out
