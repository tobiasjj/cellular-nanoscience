#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# py_fluoracle, functions that help to work with csv files stored with the Fluoracle Software
# Copyright 2018 Tobias Jachowski
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
"""
Examples
--------
import os
import py_fluoracle as pyflu

# read in all datasets in a folder
path = '.'
datasets = []

for root, dirs, files in os.walk(path):
    # Make sure to read only files in a subfolder with name 'data'
    if 'data' in root:
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                dataset = pyflu.load_data(filepath)
                if dataset is not None:
                    datasets.append(dataset)

# Sort the datasets according to their name
datasets = pyflu.humansorted_datasets(datasets)

# Get the data of the first dataset
time = datasets[0]['data'][:,0]
signal = datasets[0]['data'][:,1]
"""
__author__ = "Tobias Jachowski"
__copyright__ = "Copyright 2018"
__credits__ = ""
__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Tobias Jachowski"
__email__ = "py_fluoracle@jachowski.de"
__status__ = "beta"

import csv
import errno
import numpy as np
import os
import re

# Define the number of header lines for different measurement types
_header_lines_type = {
    'Kinetic Scan': 9,
    'Emission Scan': 21,
    'Excitation Scan': 21
}

# Set the replacement of label of the 'Xaxis'
_replace_label = {
    'Time': 'Time (s)',
    'Wavelength': 'Wavelength (nm)'
}


def load_data(filename):
    """Read in csv files saved with Fluoracle Software

    Parameters
    ----------
    filename : str
        Path of csv file to be loaded

    Returns
    -------
    dict
        dataset with the following content:
        {
            'filename': str,
            'dirname': str,
            'name': str,
            'meta': dict,
            'data': np.ndarray of type float
        }
    """
    absname = os.path.abspath(filename)
    filename = os.path.basename(absname)
    dirname = os.path.dirname(absname)

    # Number of header lines to be used to find the key 'Type'
    header_lines = 2

    # read in metadata from the header of the measurement
    meta = {}
    with open(absname, 'r') as f:
        reader = csv.reader(f)
        r = 0
        for row in reader:

            r += 1
            if len(row) >= 2:
                # Read key and correspodning value
                key = row[0]
                value = row[1]

                # Set number of header lines according to type of measurement
                if key == 'Type':
                    header_lines = _header_lines_type[value]

                # Make only first character capital
                if key == 'XAxis' or key == 'YAxis':
                    key = key.capitalize()

                # Try to convert value to a float
                try:
                    value = float(value)
                except:
                    pass

                # Store metadata
                meta[key] = value

            if r >= header_lines:
                if header_lines <= 2:
                    print('Type of file {} unknown!'.format(absname))
                    return
                break

    # read in the data
    #data = np.loadtxt(filepath, skiprows=5)
    data = np.genfromtxt(absname, delimiter=',', skip_header=header_lines, invalid_raise=False, usecols=[0,1])

    # Convert ns to s
    if meta['Xaxis'] == 'Time':
        data[:, 0] *= 1e-9

    # Change label of Xaxis
    meta['Xaxis'] = _replace_label[meta['Xaxis']]

    dataset = {
        'filename': filename,
        'dirname': dirname,
        'name': filename.strip('.csv'),
        'meta': meta,
        'data': data
    }

    return dataset


def max_fluorescence(data, sigma=0.05):
    """Determine the median of maximum fluorescence for a given sigma

    Parameters
    ----------
    data : np.ndarray of type float
        The data the maximum should be determined from
    sigma : float
        The interval the median from the maximum should be calculated from
        in fractions of 1.

    Returns
    -------
    float
        The median of the maximum signal
    """
    min_y = data.max() * (1 - sigma)
    max_y = data.max() * (1 + sigma)
    idx = np.logical_and(data > min_y, data < max_y)
    start = np.argwhere(idx).min()
    stop = np.argwhere(idx).max() + 1
    F0 = np.median(data[start:stop])
    return F0


def decay_region(data, F0=None, delay_sigma=0.95, dilution=2):
    """Autodetect the start and stop indices of a decay region

    Parameters
    ----------
    data : np.ndarray of type float
        The data the decay region should be detected from
    F0 : float (optional)
        The maximum signal / resting signal of the data
    dilution : float
        The ratio the original solution (i.e. maximum signal) is diluted
        (i.e. reduced) upon adding the reactants which lead to the decay
        of the signal.

    Returns
    -------
    int, int
        The start and stop index, respectively, of the decay region
    """
    F0 = F0 or max_fluorescence(data)

    # Determine the first fluorescence value after the fluorescence has bin diluted
    min_start = np.argwhere(data >= F0).max()
    start = np.argwhere(
        np.logical_and(data[min_start:] > F0/dilution * delay_sigma,
                       data[min_start:] <= F0/dilution)
                      ).min() + min_start
    stop = np.argwhere(data != 0).max() + 1
    return start, stop


def humansorted_strings(l):
    """Sort a list of strings like a human

    Parameters
    ----------
    l : list
        A list of strings

    Returns
    -------
    list
        The sorted list
    """
    def alphanum_key(s):
        key = re.split(r'(\d+)', s)
        key[1::2] = map(int, key[1::2])
        return key
    return sorted(l, key=alphanum_key)


def humansorted_datasets(l, key=None):
    """Sort a list of datasets according to a key of a dataset

    Parameters
    ----------
    l : list
        The list of datasets to be sorted
    key : str (optional)
        The key of the dataset the datasets should be sorted according to.
        Defaults to 'name'.

    Returns
    -------
    list
        The sorted list of datasets.
    """
    key = key or 'name'
    def alphanum_key(s):
        key = re.split(r'(\d+)', s)
        key[1::2] = map(int, key[1::2])
        return key
    def alphanum_dataset(d):
        s = d[key]
        return alphanum_key(s)
    return sorted(l, key=alphanum_dataset)


def mkdir(directory):
    """Create a directory only if it exists

    Parameters
    ----------
    directory : str
        The directory to be created.
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
