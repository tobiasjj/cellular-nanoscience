#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# files, functions to browse and display files
# Copyright 2016, 2017, 2018, 2019 Tobias Jachowski
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

import os
from ipywidgets import interact, IntSlider
from IPython.core.display import Image, display

def file_and_dir(filename=None, directory=None):
    filename = filename or ""
    fdir = os.path.dirname(filename)
    ffile = os.path.basename(filename)

    ddir = directory or "."

    if (ffile == "" or ffile == "." or ffile == ".."):
        directory = os.path.join(ddir, filename, "")
        absdir = os.path.realpath(directory)
        return None, absdir, None

    directory = os.path.join(ddir, fdir, "")
    absdir = os.path.realpath(directory)
    absfile = os.path.join(absdir, ffile)

    return ffile, absdir, absfile


def files(directory, prefix=None, suffix=None, extension=None, sort_key=None):
    """
    Get filenames of a directory in the order sorted to their filename or a
    given key function.

    Parameters
    ----------
    directory : str
        The directory the files are located in.
    prefix : str
        Get only the files beginning with `prefix`.
    suffix : str
        Get only the files ending with `suffix`.
    extension : str, optional
        The extension of the files that should be returned. Default is
        '.txt'.
    sort_key : function
        Function to be applied to every filename found, before sorting.
    """
    prefix = prefix or ''
    suffix = suffix or ''
    extension = extension or ''
    files = [file_and_dir(filename=name, directory=directory)[2]
             for name in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, name))
             and name.startswith(prefix)
             and name.endswith(''.join((suffix, extension)))]
    files.sort(key=sort_key)
    return files


def browse_images(directory, prefix=None, suffix=None, extension='.png',
                  sort_key=None):
    """
    Display images of a directory with an interactive widget and a slider in a
    jupyter notebook, in the order sorted to their filename or a given key
    function.

    Parameters
    ----------
    directory : str
        The directory the images to be displayed are located in.
    prefix : str
        Display only the files beginning with `prefix`.
    suffix : str
        Display only the files ending with `suffix`.
    extension : str, optional
        The extension of the images that should be displayed. Default is
        '.png'.
    sort_key : function
        Function to be applied to every image filename found, before sorting.
    """
    images = files(directory, prefix, suffix, extension, sort_key)
    stop = len(images)
    if stop < 1:
        print("No file found with prefix '%s', suffix '%s', and extension '%s'"
              % (prefix, suffix, extension))
        return

    def view_image(i):
        image = images[i]
        try:
            display(Image(filename=image))
            print(image)
        except:
            print('No image found!')

    slider = IntSlider(min=0, max=stop-1, step=1, value=0,
                       description='Image:')

    return interact(view_image, i=slider)
