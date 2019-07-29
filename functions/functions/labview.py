#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# labview, functions to read and write labview binary data
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
import struct


def chunk_info(filename):
    """
    Read all chunk arrays with the information (number of rows and columns) of
    the data, following each chunk.
    """
    with open(filename, "rb") as f:
        pos = 0
        while True:
            byte = f.read(8)
            pos += 1
            if byte:
                chunk_shape = struct.unpack('>2i', byte)
                if chunk_shape[0] * chunk_shape[1] > 0:
                    # go to byte offset relative to current strem position
                    f.seek(8 * chunk_shape[0] * chunk_shape[1], 1)
                    yield(pos, *chunk_shape)
                    pos += chunk_shape[0] * chunk_shape[1]
            else:
                break


def chunk_data(filename, chunks, dtype='>d'):
    with open(filename, "rb") as f:
        for chunk in chunks:
            # go to byte offset where chunk starts
            f.seek(chunk[0] * 8)
            # read the data
            data_bin = f.read(chunk[1] * chunk[2] * 8)
            data = np.fromstring(data_bin, dtype=dtype)
            yield data


def read_labview_bin_data(filename, dtype='>d', start_row_idx=0,
                          number_of_rows=-1):
    """
    Parameters
    ----------
    filename : str
        Path of the binary file to read
    dtype : str
        Type of double
    start_row_idx : int
        Index of the first datapoint (of all traces) to read
    number_or_rows : int
        Number of datapoints (of all traces) to read. Defaults to number of
        datapoints of the binary file - `start_row_idx`
    """
    stop_row_idx = start_row_idx + number_of_rows  # index of row to stop read
    rows_to_read = 0  # rows to read after getting information of chunks
    columns = None  # number of columns (i.e. traces) in binary file
    chunks = []  # chunks with the index information of data to be read
    chunk_row_start = 0  # running index of first row of chunk
    chunk_row_stop = 0  # running stop index of last row of chunk

    if number_of_rows == 0:
        return np.empty((0, 0))

    print('Getting chunk info from:')
    print('  \'%s\'' % filename)
    for chunk in chunk_info(filename):
        # Set the running indices to the new position (row)
        chunk_row_start = chunk_row_stop
        chunk_row_stop += chunk[1]
        # Check if information about number of columns changed from one to the
        # next chunk
        if columns == chunk[2] or columns is None:
            columns = chunk[2]
        else:
            print("Number of columns in chunks of file differ from each other!")
        # Check if chunk is (partly) contained within the requested data index
        if (chunk_row_stop > start_row_idx
                and (number_of_rows < 0 or chunk_row_start < stop_row_idx)):
            # Check read position and number of rows of first chunk
            if len(chunks) is 0:
                shift = start_row_idx - chunk_row_start
                chunk = (chunk[0] + shift * chunk[2],
                         chunk[1] - shift,
                         chunk[2])
            chunks.append(chunk)
            rows_to_read += chunk[1]
        # Check stop position and number of rows of last chunk
        if (number_of_rows > 0 and chunk_row_stop >= stop_row_idx):
            shift = chunk_row_stop - stop_row_idx
            chunk = (chunk[0], chunk[1] - shift, chunk[2])
            chunks[-1] = chunk
            rows_to_read -= shift
            break

    if rows_to_read == 0:
        return np.empty((0, 0))

    print('Reading data chunks from:')
    print('  \'%s\'' % filename)
    data = np.empty(rows_to_read * columns)
    i = 0
    for _data in chunk_data(filename, chunks, dtype=dtype):
        length = len(_data)
        data[i:i + length] = _data
        i += length

    data = data.reshape(rows_to_read, columns)

    # Workaround, if last column is 0.0
    if np.all(data[:, -1] == 0.0):
        # kick out last column
        data = data[:, :-1]

    return data


def write_labview_bin_data(filename, array, dtype='d', endianness='>'):

    i = struct.pack('>i', array.shape[0])
    j = struct.pack('>i', array.shape[1])
    fmt = endianness + dtype * array.size
    data = struct.pack(fmt, *array.flatten())

    with open(filename, 'wb') as f:
        f.write(i)
        f.write(j)
        f.write(data)
'''
def convert_bin_data(fname_i, fname_o, chunk_size, chunk_shape_size=1):
    """
    Remove chunk size information of labview binary files.
    """
    fi = open(fname_i, 'rb')
    fo = open(fname_o, 'wb')

    while True:
        buf = fi.read(chunk_size)
        if len(buf) == 0:
            break
        fo.write(buf[chunk_shape_size:])

        fi.close()
        fo.close()
'''
