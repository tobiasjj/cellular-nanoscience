#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Create videos, create videos from image files
# Copyright 2018, 2019 Tobias Jachowski
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

import imageio
import numpy as np
import os
import pims
import re
import tifffile

from ipywidgets import (HBox, VBox, BoundedFloatText, BoundedIntText,
                        interactive, IntRangeSlider, IntSlider, Text)
from IPython.display import display
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def check_set(para, default, decider=None):
    if decider is None:
        decider = para
    para = default if decider is None else para
    return para


def get_filenames(directory):
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            yield name


def filterby(names, prefix=None, suffix=None, extension=None):
    prefix = check_set(prefix, '')
    suffix = check_set(suffix, '')
    extension = check_set(extension, '')
    for name in names:
        if (name.startswith(prefix)
                and name.endswith(''.join((suffix, extension)))):
            yield name


def file_and_dir(filename=None, directory=None):
    filename = check_set(filename, '')
    fdir = os.path.dirname(filename)
    ffile = os.path.basename(filename)

    ddir = check_set(directory, '.')

    if (ffile == '' or ffile == '.' or ffile == '..'):
        directory = os.path.join(ddir, filename, '')
        absdir = os.path.realpath(directory)
        return None, absdir, None

    directory = os.path.join(ddir, fdir, '')
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
        The extension of the files that should be returned.
    sort_key : function
        Function to be applied to every filename found, before sorting.
    """
    names = get_filenames(directory)
    names_filtered = filterby(names, prefix=prefix, suffix=suffix,
                              extension=extension)
    fullnames = [file_and_dir(filename=name, directory=directory)[2]
                 for name in names_filtered]
    fullnames.sort(key=sort_key)
    return fullnames


def split_filenames(fullnames, split_time=None, regexp=None):
    split_time = check_set(split_time, 0.0)
    if split_time > 0.0:
        if regexp is None:
            times = np.array([creation_time(f) for f in fullnames])
            tdiffs = times[1:] - times[:-1]
        else:
            try:
                # Extract time difference from filename
                # The tdiff in each filename represents the time passed since
                # the previous image was taken
                # example name: 'B14-029ms-8902.tif'
                # example regexp: '-([0-9]*)ms-'
                tdiffs = np.array(
                    [int(re.findall(regexp, os.path.basename(filename))[0])
                     for filename in fullnames])[1:] / 1000
            except:
                tdiffs = np.zeros(len(fullnames) - 1)
        idx = np.r_[0, np.where(tdiffs >= split_time)[0] + 1, len(fullnames)]
        start_stop_idx = np.c_[idx[:-1], idx[1:]]
    else:
        start_stop_idx = np.array([[0, len(fullnames)]])

    filelists = []
    for start, stop in start_stop_idx:
        filelists.append((fullnames[start:stop], start, stop))
    return filelists


def dtype_info(dtype=None, array=None):
    if array is not None:
        dtype = check_set(dtype, array.dtype)
    try:
        info = np.iinfo(dtype)
    except:
        info = np.finfo(dtype)
    return dtype, info


def lookup_table(dtype, minimum=None, maximum=None, dtype_to=None):
    dtype, info = dtype_info(dtype=dtype)
    dtype_to = check_set(dtype_to, dtype)
    dtype_to, info_to = dtype_info(dtype=dtype_to)

    minimum = check_set(minimum, info.min)
    maximum = check_set(maximum, info.max)
    minimum = max(minimum, info.min)
    maximum = min(maximum, info.max)

    # Create a lookup table
    lut = np.zeros(2**info.bits, dtype=dtype_to)
    lut[:minimum] = info_to.min
    lut[minimum:maximum + 1] = np.linspace(info_to.min, info_to.max,
                                           maximum - minimum + 1,
                                           dtype=dtype_to)
    lut[maximum:] = info_to.max

    return lut


def adjust_contrast(image, minimum, maximum):
    dtype = image.dtype
    lut = lookup_table(dtype, minimum, maximum)
    return np.take(lut, image)


def convert_image(image, dtype_to, minimum, maximum):
    dtype = image.dtype
    lut = lookup_table(dtype, minimum, maximum, dtype_to)
    return np.take(lut, image)


def get_minmax_grey(image, minimum=None, maximum=None, width=None,
                    offset=None):
    if width is None:
        minimum = check_set(minimum, image.min())
        maximum = check_set(maximum, image.max())
    else:
        # adjust histogramm according to given width, centered around the
        # median
        #hist, values = np.histogram(image, bins=2**16-1)
        #middle = hist.argmax()
        median = np.median(image)
        offset = check_set(offset, 0)
        minimum = median - width / 2 + offset
        maximum = minimum + width
    minimum = int(round(minimum))
    maximum = int(round(maximum))
    return minimum, maximum


def convert_uint16_uint8(image, minimum=None, maximum=None, width=None,
                         offset=None):
    minimum, maximum = get_minmax_grey(image, minimum, maximum, width, offset)
    return convert_image(image, 'uint8', minimum, maximum)


def calculate_background(filenames, defect_roi=None, replace_mode='v'):
    """
    Parameters
    ----------
    defect_roi : (float, float, float, float)
        Tuple of (center_x, center_y, length_x, length_y)
    replace_mode : str
        Replace mode of either vertical ('v') or horizontal ('h')
    """
    # Calculate background image from median of files
    tiffstack = tifffile.imread(filenames)
    median = np.median(tiffstack, axis=0).astype('uint16')

    # Replace defect regions - e.g. from the trapping of beads - with
    # neighbouring pixels
    if defect_roi is not None:
        height, width = median.shape
        start_x, stop_x, start_y, stop_y = get_crop_image_roi(width, height,
                                                              *defect_roi)

        if replace_mode == 'v':  # vertical replacement
            # length_y pixel
            px = int(np.ceil(defect_roi[3] / 2))
            # replace bottom half
            median[start_y:stop_y-px, start_x:stop_x] = \
                median[start_y-px:start_y, start_x:stop_x]
            # replace top half
            median[start_y+px:stop_y, start_x:stop_x] = \
                median[stop_y:stop_y+px, start_x:stop_x]
        else:  # horizontal replacement
            # length_x pixel
            px = int(np.ceil(defect_roi[2] / 2))
            # replace left half
            median[start_y:stop_y, start_x:stop_x - px] = \
                median[start_y:stop_y, start_x - px:start_x]
            # replace right half
            median[start_y:stop_y, start_x + px:stop_x] = \
                median[start_y:stop_y, stop_x:stop_x + px]

    # normalize the median to the minimum
    minimum = median.min()
    return median - minimum


def bin_image(image, bin_by=None, func=None):
    """
    image : np.ndarray 2D
    bin_by : iterable 2D
    func : function
           function to apply to each 2D bin. Defaults to `np.mean`.
    """
    if bin_by is None:
        return image
    func = check_set(func, np.mean)
    shape = np.asarray(image.shape)
    bin_by = np.asarray(bin_by[::-1])
    shape_new = shape // bin_by
    factor = (np.asarray(shape) // shape_new)

    image_binned = func(func(
        image.reshape(shape_new[0], factor[0],
                      shape_new[1], factor[1]), 1), 2)

    return image_binned.astype(image.dtype)


def annotate(image, pos, size, text, font=None):
    # Get max color of image
    dtype, info = dtype_info(array=image)
    fill = int(255 / info.max * image.max())

    font = check_set(font, 'FreeSans')
    font = ImageFont.truetype(font, size=size)

    # Convert array to Image
    image = Image.fromarray(image)

    # Draw the text
    draw = ImageDraw.Draw(image)
    draw.text(pos, text, fill=fill, font=font)

    return np.array(image)


class adjust_roi(object):
    def __init__(self, image=None):
        """
        Parameters
        ----------
        image : np.ndarray
        """
        if image is not None:
            self.length_y, self.length_x = image.shape
            self.center_x = self.length_x / 2
            self.center_y = self.length_y / 2
        else:
            self.center_x = None
            self.center_y = None
            self.length_x = None
            self.length_y = None

        self.fig, axes = plt.subplots(ncols=2)
        self.axes = axes.flatten()

        self.image = None
        if image is not None:
            self.process_image(image)

    def __call__(self, image=None):
        self.process_image(image)

    def process_image(self, image=None):
        if image is not None:
            self.image = image
        height, width = self.image.shape
        crop_roi = get_crop_image_roi(width, height,
                                      self.center_x, self.center_y,
                                      self.length_x, self.length_y)
        start_x, stop_x, start_y, stop_y = crop_roi
        if image is not None:
            self.axes[0].clear()
            self.axes[0].imshow(image, cmap=plt.cm.gray)
            self.rectangleselector = RectangleSelector(self.axes[0],
                                                       self.set_roi,
                                                       useblit=True,
                                                       interactive=False)
            self.rectangle = patches.Rectangle((start_x, start_y),
                                               stop_x - start_x,
                                               stop_y - start_y,
                                               linewidth=1, edgecolor='r',
                                               fill=False)
            self.axes[0].add_patch(self.rectangle)
        self.axes[1].clear()
        im_c = self.image[start_y:stop_y, start_x:stop_x]
        self.axes[1].imshow(im_c, cmap=plt.cm.gray)
        return im_c

    def set_roi(self, pos_1, pos_2):
        """
        Set the ROI and update the figure accordingly.
        This method is called upon any change of the RectangleSelector.

        Parameters
        ----------
        pos_1 : (float, float)
            Position of one corner of the rectangle (x1, y1)
        pos_2 : (float, float)
            Position of the diagonally opposite corner of the rectangle (x2,
            y2)
        """
        try:
            x1, y1, x2, y2 = pos_1.xdata, pos_1.ydata, pos_2.xdata, pos_2.ydata
        except:
            x1, y1, x2, y2 = pos_1[0], pos_1[1], pos_2[0], pos_2[1]

        height, width = self.image.shape
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
        # round to .5 (center between two pxs)
        self.center_x = round(0.5 * round(x1 + x2), 1)
        self.center_y = round(0.5 * round(y1 + y2), 1)
        self.length_x = int(round(abs(x2 - x1))) + 1
        self.length_y = int(round(abs(y2 - y1))) + 1

        # Update the rectangle of the original image plot
        crop_roi = get_crop_image_roi(width, height,
                                      self.center_x, self.center_y,
                                      self.length_x, self.length_y)
        start_x, stop_x, start_y, stop_y = crop_roi
        self.rectangle.set_xy((start_x, start_y))
        self.rectangle.set_width(self.length_x)
        self.rectangle.set_height(self.length_y)

        # Process the image
        self.process_image()


class adjust_image_contrast(object):
    def __init__(self, image=None, dtype=None):
        """
        Display images of a directory with an interactive widget and a slider
        in a jupyter notebook, in the order sorted to their filename or a given
        key function.

        Parameters
        ----------
        image : np.ndarray
        dtype : np.ndarray.dtype
        """
        if dtype is not None or image is not None:
            dtype, info = dtype_info(array=image, dtype=dtype)
            self.min = info.min
            self.max = info.max
        else:
            self.min = None
            self.max = None

        self.fig, axes = plt.subplots(nrows=2, ncols=2)
        self.axes = axes.flatten()

        self.image = None
        if image is not None:
            self.process_image(image)

    def __call__(self, image=None):
        self.process_image(image)

    def process_image(self, image=None):
        if image is not None:  # new image
            self.image = image
            self.axes[0].clear()
            self.axes[0].imshow(image, cmap=plt.cm.gray)
            self.axes[2].clear()
            # plot hist with 8bit bins
            self.axes[2].hist(image.ravel(), bins=(2**8))
            self.spanselector = SpanSelector(self.axes[2],
                                             self.set_minmax_grey,
                                             'horizontal',
                                             useblit=True)
            self.axspan = self.axes[2].axvspan(self.min, self.max,
                                               facecolor='y', alpha=0.2)
        self.axes[1].clear()
        self.axes[3].clear()
        self.image_contrast = adjust_contrast(self.image, self.min, self.max)
        im_c = self.image_contrast
        self.axes[1].imshow(im_c, cmap=plt.cm.gray)
        self.axes[3].hist(im_c.ravel(), bins=(2**8))
        return self.image_contrast

    def set_minmax_grey(self, min_grey, max_grey):
        """
        Set the grey min max values and update the figure accordingly.
        This function is called upon any change of the SpanSelector.

        Parameters
        ----------
        min_grey : float
        max_grey : float
        """
        self.min_grey = min(max(int(np.round(min_grey)), 0), 2**16)
        self.max_grey = min(max(int(np.round(max_grey)), 0), 2**16)
        try:
            dtype, info = dtype_info(array=self.image)
            self.min_grey = max(self.min_grey, info.min)
            self.max_grey = min(self.max_grey, info.max)
        except:
            pass

        # Update the axvspan of the original histogram plot
        self.axspan.set_xy([[self.min_grey, 0],  # lower left corner
                            [self.min_grey, 1],  # upper left corner
                            [self.max_grey, 1],  # upper right corner
                            [self.max_grey, 0],  # lower right corner
                            [self.min_grey, 0]])  # lower left corner

        # Process the image
        self.process_image()


class adjust_roi_contrast(object):
    def __init__(self, image=None, dtype=None,
                 background_image=None, min_grey=None, max_grey=None):
        """
        Display images of a directory with an interactive widget and a slider
        in a jupyter notebook, in the order sorted to their filename or a given
        key function.

        Parameters
        ----------
        image : np.ndarray
        dtype : np.ndarray.dtype
        """
        # The variable self.image gets set in the function self.process_image()
        # (see further down).
        self.image = image
        self.image_processed = None
        self.image_background = background_image
        if background_image is not None:
            if isinstance(background_image, str):
                self.image_background = pims.open(background_image)[0]

        # Get default ROI values to crop the image
        if self.image is not None:
            length_y, length_x = self.image.shape
            center_x = length_x / 2
            center_y = length_y / 2
        else:
            center_x = 0.0
            center_y = 0.0
            length_x = 1
            length_y = 1

        # Get default minmax grey levels
        if dtype is not None or self.image is not None:
            dtype, info = dtype_info(array=self.image, dtype=dtype)
            min_grey = info.min
            max_grey = info.max
        else:
            min_grey = 0
            max_grey = 2**16 - 1  # assume 16bit image

        # Create plots to display the original image, the histograms and the
        # cropped and contrast/brightness corrected image
        self.fig, axes = plt.subplots(nrows=2, ncols=2)
        self.axes = axes.flatten()

        # Create widgets for storing and setting the ROI values with
        # callback function to trigger the update of the ROI
        self._center_x = BoundedFloatText(value=center_x,
                                          min=0, max=length_x - 1,
                                          step=0.5, description='center_x:')
        self._center_y = BoundedFloatText(value=center_y,
                                          min=0, max=length_y - 1,
                                          step=0.5, description='center_y:')
        self._length_x = BoundedIntText(value=length_x,
                                        min=1, max=length_x,
                                        step=1, description='length_x:')
        self._length_y = BoundedIntText(value=length_y,
                                        min=1, max=length_y,
                                        step=1, description='length_y:')
        left_box = VBox([self._center_x, self._center_y])
        right_box = VBox([self._length_x, self._length_y])
        ui_crop = HBox([left_box, right_box])

        self.stop_roi_callback = False

        def _set_roi(value):
            if self.stop_roi_callback:
                return
            if self.image is not None:
                height, width = self.image.shape
                crop_roi = get_crop_image_roi(width, height,
                                              self.center_x, self.center_y,
                                              self.length_x, self.length_y)
                start_x, stop_x, start_y, stop_y = crop_roi
                pos_1 = (start_x, start_y)
                pos_2 = (stop_x - 1, stop_y - 1)
                self.set_roi(pos_1, pos_2, update_widgets=False)
        self.center_x_interact = interactive(_set_roi, value=self._center_x)
        self.center_y_interact = interactive(_set_roi, value=self._center_y)
        self.length_x_interact = interactive(_set_roi, value=self._length_x)
        self.length_y_interact = interactive(_set_roi, value=self._length_y)

        # Create widgets for storing and setting the histogram min/max grey
        # levels and callback function to trigger the update of histogram
        self._min_grey = BoundedIntText(value=min_grey,
                                        min=min_grey, max=max_grey,
                                        step=1, description='min_grey:')
        self._max_grey = BoundedIntText(value=max_grey,
                                        min=min_grey, max=max_grey,
                                        step=1, description='max_grey:')
        ui_grey = VBox([self._min_grey, self._max_grey])
        display(ui_crop)
        display(ui_grey)

        self.stop_grey_callback = False

        def _set_minmax_grey(value):
            if self.stop_grey_callback:
                return
            if self.min_grey > self.max_grey:
                self.stop_grey_callback = True
                if value == self.min_grey:
                    self._min_grey.value = self.max_grey
                else:
                    self._max_grey.value = self.min_grey
                self.stop_grey_callback = False

            if self.image is not None:
                self.set_minmax_grey(self.min_grey, self.max_grey,
                                     update_widgets=False)
        self.center_x_interact = interactive(_set_minmax_grey,
                                             value=self._min_grey)
        self.center_y_interact = interactive(_set_minmax_grey,
                                             value=self._max_grey)

        # Call process_image only if an image was provided. Otherwise,
        # the function would fail due to uninitialized self.image
        if image is not None:
            self.process_image(image)

    def __call__(self, image=None):
        self.process_image(image)

    def process_image(self, image=None):
        """
        Is called if the image or a process parameter of the image is changed.
        """
        # A new image is loaded. Store the new image.
        if image is not None:
            self.image = image

        # Check, if the dimensions of the image have changed. If so, initialize
        # the widgets with the parameters of the new image.
        height, width = self.image.shape
        if width == self._length_x.max and height == self._length_y.max:
            # Image dimensions have not changed.
            center_x = self.center_x
            center_y = self.center_y
            length_x = self.length_x
            length_y = self.length_y
        else:
            # Image dimensions have changed
            center_x = (width - 1) / 2
            center_y = (height - 1) / 2
            length_x = width
            length_y = height
            self.stop_roi_callback = True
            self._center_x.max = width - 1
            self._center_y.max = height - 1
            self._length_y.max = height
            self._length_x.max = width
            self._center_x.value = (width - 1) / 2
            self._center_y.value = (height - 1) / 2
            self._length_x.value = width
            self._length_y.value = height
            self.stop_roi_callback = False

        # Calculate the ROI to crop the image
        crop_roi = get_crop_image_roi(width, height,
                                      center_x, center_y,
                                      length_x, length_y)
        start_x, stop_x, start_y, stop_y = crop_roi

        # Check if the dtype of the image has changed and initialize the
        # histogram widgets accordingly
        dtype, info = dtype_info(array=self.image)
        if info.min != self._min_grey.min or info.max != self._max_grey.max:
            self.stop_grey_callback = True
            self._min_grey.min = info.min
            self._min_grey.max = info.max
            self._max_grey.min = info.min
            self._max_grey.max = info.max
            self.stop_grey_callback = False

        # Get the background corrected original image
        image_process = self.image_background_corrected

        # Update the plot of the new background corrected image
        if image is not None:
            self.axes[0].clear()
            self.axes[0].imshow(image_process, cmap=plt.cm.gray)
            self.rectangleselector = RectangleSelector(self.axes[0],
                                                       self.set_roi,
                                                       useblit=True,
                                                       interactive=False)
            self.rectangle = patches.Rectangle((start_x, start_y),
                                               stop_x - start_x,
                                               stop_y - start_y,
                                               linewidth=1, edgecolor='r',
                                               fill=False)
            self.axes[0].add_patch(self.rectangle)

        # ROI crop the original image
        image_process = image_process[start_y:stop_y, start_x:stop_x]

        # Update the histogram of the cropped original image
        self.axes[2].clear()
        # plot hist with 8bit bins
        self.axes[2].hist(image_process.ravel(), bins=(2**8))
        self.axes[2].set_yscale('log')

        self.spanselector = SpanSelector(self.axes[2], self.set_minmax_grey,
                                         'horizontal', useblit=True)

        # Update the axvspan with the current min_grey/max_grey levels
        if (self.min_grey != self._min_grey.min
                or self.max_grey != self._max_grey.max):
            self.axspan = self.axes[2].axvspan(self.min_grey, self.max_grey,
                                               facecolor='y', alpha=0.2)

        # Adjust the contrast of the cropped image
        image_process = convert_image(image_process, 'uint8',
                                      self.min_grey, self.max_grey)

        # Update the plot of the processed image
        self.axes[1].clear()
        self.axes[1].imshow(image_process, cmap=plt.cm.gray)

        # Update the histogram of the processed image
        self.axes[3].clear()
        # plot hist with 8bit bins
        self.axes[3].hist(image_process.ravel(), bins=(2**8))
        self.axes[3].set_yscale('log')

        self.image_processed = image_process
        return image_process

    def set_roi(self, pos_1, pos_2, update_widgets=True):
        """
        Set the ROI and update the figure accordingly.
        This method is called upon any change of the RectangleSelector.

        Parameters
        ----------
        pos_1 : (float, float)
            Position of one corner of the rectangle (x1, y1)
        pos_2 : (float, float)
            Position of the diagonally opposite corner of the rectangle (x2,
            y2)
        """
        try:
            x1, y1, x2, y2 = pos_1.xdata, pos_1.ydata, pos_2.xdata, pos_2.ydata
        except:
            x1, y1, x2, y2 = pos_1[0], pos_1[1], pos_2[0], pos_2[1]

        height, width = self.image.shape

        # Update interact widget values according to new values, only
        # if set_roi was not called from a widget interaction itself
        if update_widgets:
            x1 = min(max(x1, 0), width - 1)
            y1 = min(max(y1, 0), height - 1)
            x2 = min(max(x2, 0), width - 1)
            y2 = min(max(y2, 0), height - 1)
            # round to .5 (in between tow pxs)
            center_x = round(0.5 * round(x1 + x2), 1)
            center_y = round(0.5 * round(y1 + y2), 1)
            length_x = int(round(abs(x2 - x1))) + 1
            length_y = int(round(abs(y2 - y1))) + 1
            self.stop_roi_callback = True
            self._center_x.value = center_x
            self._center_y.value = center_y
            self._length_x.value = length_x
            self._length_y.value = length_y
            self.stop_roi_callback = False

        # Update the rectangle of the original image plot
        crop_roi = get_crop_image_roi(width, height,
                                      self.center_x, self.center_y,
                                      self.length_x, self.length_y)
        start_x, stop_x, start_y, stop_y = crop_roi
        self.rectangle.set_xy((start_x, start_y))
        self.rectangle.set_width(self.length_x)
        self.rectangle.set_height(self.length_y)

        # Process the image
        self.process_image()

    def set_minmax_grey(self, min_grey, max_grey, update_widgets=True):
        """
        Set the grey min max levels and update the figure accordingly.
        This function is called upon any change of the SpanSelector.

        Parameters
        ----------
        min_grey : float
        max_grey : float
        """
        min_grey = min(max(int(np.round(min_grey)), 0), 2**16)
        max_grey = min(max(int(np.round(max_grey)), 0), 2**16)
        try:
            dtype, info = dtype_info(array=self.image)
            min_grey = max(min_grey, info.min)
            max_grey = min(max_grey, info.max)
        except:
            pass

        # Update interact widget values according to new values, only
        # if set_minmax_grey was not called from a widget interaction itself
        if update_widgets:
            self.stop_grey_callback = True
            self._min_grey.value = min_grey
            self._max_grey.value = max_grey
            self.stop_grey_callback = False

        # Process the image
        self.process_image()

    @property
    def image_background_corrected(self):
        if self.image is not None:
            background = 0
            if self.image_background is not None:
                background = self.image_background
            image = self.image - background
        return image

    @property
    def center_x(self):
        return self._center_x.value

    @property
    def center_y(self):
        return self._center_y.value

    @property
    def length_x(self):
        return self._length_x.value

    @property
    def length_y(self):
        return self._length_y.value

    @property
    def min_grey(self):
        return self._min_grey.value

    @property
    def max_grey(self):
        return self._max_grey.value


class process_images(object):
    def __init__(self, process, directory, prefix=None, suffix=None,
                 extension='.tif', sort_key=None):
        """
        Display images of a directory with an interactive widget and a slider
        in a jupyter notebook, in the order sorted to their filename or a given
        key function.

        Parameters
        ----------
        process : function
            function, which takes an image (np.ndarray) as an argument
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
            Function to be applied to every image filename found, before
            sorting.
        """
        sort_key = check_set(sort_key, creation_time)
        self.filenames = files(directory, prefix, suffix, extension, sort_key)
        self.filelists = [(start, stop) for filenames, start, stop
                          in split_filenames(self.filenames)]
        self.process = process

        stop = len(self.filenames)
        if stop < 1:
            print("No file found with prefix '%s', suffix '%s', and extension '%s'"
                  % (prefix, suffix, extension))
            return
        self.split_regexp = Text(value='', placeholder='Regular Expression',
                                 description='SplitRegExp:')
        self.splittime = BoundedFloatText(value=0.0, min=0, max=5,
                                          step=0.05,
                                          description='SplitTime (s):')
        self.listselect = IntSlider(value=0, min=0, max=len(self.filelists)-1,
                                    step=1, description='MovieIDX:')
        self.filerange = IntRangeSlider(value=[0, stop-1], min=0, max=stop-1,
                                        step=1, description='ImagesIDX:')
        self.fileselect = IntSlider(value=0, min=0, max=stop-1,
                                    step=1, description='ImageIDX:')
        left_box = VBox([self.split_regexp, self.splittime])
        right_box = VBox([self.listselect, self.filerange, self.fileselect])
        ui = HBox([left_box, right_box])
        display(ui)

        def set_splittime(t, regexp):
            if regexp == '':
                regexp = None
            self.filelists = [(start, stop) for filenames, start, stop
                              in split_filenames(self.filenames, split_time=t,
                                                 regexp=regexp)]
            self.listselect.max = len(self.filelists) - 1
            set_filelist(self.listselect.value)

        def set_filelist(i):
            min = self.filelists[i][0]
            max = self.filelists[i][1] - 1
            if max < self.filerange.min:
                self.filerange.min = min
                self.filerange.max = max
            else:  # new min > old max
                self.filerange.max = max
                self.filerange.min = min
            self.filerange.value = (min, max)

        def set_range(minmax):
            min = minmax[0]
            max = minmax[1]
            if max < self.fileselect.min:
                self.fileselect.min = min
                self.fileselect.max = max
            else:
                self.fileselect.max = max
                self.fileselect.min = min

        def process_image(i):
            self.process_image(i)
        self.splitinteract = interactive(set_splittime, t=self.splittime,
                                         regexp=self.split_regexp)
        self.listinteract = interactive(set_filelist, i=self.listselect)
        self.rangeinteract = interactive(set_range, minmax=self.filerange)
        self.selectinteract = interactive(process_image, i=self.fileselect)

        # Trigger first update
        if len(self.filenames) > 0:
            self.process_image(0)

    def process_image(self, i):
        filename = self.filenames[i]
        try:
            print(filename)
            image = pims.open(filename)[0]
        except:
            print('No image found!')
        self.process(image)

    @property
    def split_time(self):
        return self.splittime.value

    @property
    def min_idx(self):
        return self.filerange.min

    @property
    def max_idx(self):
        return self.filerange.max

    @property
    def file_idx(self):
        return self.fileselect.value


def get_image_shape(filename):
    height, width = pims.open(filename)[0].shape
    return width, height


def get_crop_image_roi(width, height, center_x=None, center_y=None,
                       length_x=None, length_y=None, multiple_of=None):
    def get_start_stop(length, center, new_length, multiple_of=None):
        if center is None:
            center = (length - 1) / 2  # middle of the image
        # round to .5 (in between two pixels is allowed)
        else: round(0.5 * round(center * 2), 1)
        new_length = int(round(check_set(new_length, length)))

        # center between 0 and length - 1 (first/last px)
        center = min(max(center, 0), length - 1)
        # set max_length according to position of center
        max_length = int(round(min(center, length - center - 1) * 2 + 1))
        # set new_length between 1 and (new_length or max_length)
        new_length = min(max(new_length, 1), max_length)

        # make new_length a multiple of multiple_of
        if multiple_of is not None:
            multiple_of = int(round(min(max(multiple_of, 1), max_length)))
            reminder = new_length % multiple_of
            if reminder >= multiple_of / 2:
                new_length += multiple_of - reminder
            else:
                new_length -= reminder
            if new_length > max_length:
                new_length -= multiple_of
            new_length = max(new_length, 1)

        start = center - (new_length - 1) * 0.5
        start = min(max(start, 0), length - new_length)
        start = int(np.floor(start))
        stop = start + new_length
        return start, stop

    multiple_of = np.asarray(multiple_of).flatten().tolist()
    if len(multiple_of) == 1:
        multiple_of.append(multiple_of[0])
    start_x, stop_x = get_start_stop(width, center_x, length_x,
                                     multiple_of[0])
    start_y, stop_y = get_start_stop(height, center_y, length_y,
                                     multiple_of[1])

    return (start_x, stop_x, start_y, stop_y)


def scalebar(image, resolution=None, width=None, height=None,
             pos_x_rel=None, pos_y_rel=None, value=None):
    """
    Draw a scalebar on top of an image

    Parameters
    ----------
    resolution : float or list of floats
        The resolution of the image in unit/px for x and y
    width : float
        The width of the scalebar in units. Defaults to 1.0.
    height : float
        The height of the scalebar in units. Defaults to 0.15*`width`.
    pos_x_rel : float
        Relative x position of image up to where the scalebar should extend.
        Defaults to 0.98.
    pos_y_rel : float
        Relative y position of image up to where the scalebar should extend.
        Defaults to "(height - (width * (1 - pos_x_rel))) / height".
    value : int
        The integer value of the color of the scalebar. Defaults to the maximal
        allowed value of the `image` array. Depending on the lookup table this
        translates usually to white or black.
    """
    resolution = check_set(resolution, 1.0)
    resolution = np.asarray(resolution).flatten().tolist()
    if len(resolution) == 1:
        resolution.append(resolution[0])
    width = check_set(width, 1.0)
    width_px = int(round(width / resolution[0]))
    height = check_set(height, 0.15 * width)
    height_px = int(round(height / resolution[1]))

    image_height, image_width = image.shape
    pos_x_rel = check_set(pos_x_rel, 0.98)
    pos_x = int(round(pos_x_rel * image_width))
    if pos_y_rel is None:
        pos_y = image_height - (image_width - pos_x)
    else:
        pos_y = int(round(pos_y_rel * image_height))

    dtype, info = dtype_info(array=image)
    value = check_set(value, info.max)

    image[pos_y- height_px:pos_y, pos_x - width_px:pos_x] = value


def creation_time(filename):
    return os.stat(filename).st_mtime


def get_fps(filenames, fps=None):
    """
    Parameters
    ----------
    fps : float or str
        Autodetect or set the frames per second of the source files.
        'predominant' calculates the median of all creation time differences
        between all files and uses the reciprocal as fps. 'total' uses the
        difference of the creation time of the last and the first file and
        divides it by the total number of images. 'no_fps' returns `None`.
        Defaults to 'predominant'.
    """
    def fps_explicit():
        return fps
    def fps_predominant():
        times = np.array([creation_time(f) for f in filenames])
        tdiffs = times[1:] - times[:-1]
        tdiff = np.median(tdiffs)
        fps = 1/tdiff
        return fps
    def fps_total():
        start_time = creation_time(filenames[0])
        end_time = creation_time(filenames[-1])
        duration = end_time - start_time
        fps = (len(filenames) - 1) / duration
        return fps
    fps_options = {
        None: fps_predominant,
        'predominant': fps_predominant,
        'total': fps_total,
        'no_fps': None
    }

    # Determine fps ('predominant' or 'total') or set directly ('fps_explicit')
    return fps_options.get(fps, fps_explicit)()


def _create_video(filenames, savename, background_image=None, bin_by=None,
                  min_grey=None, max_grey=None, width_grey=None,
                  offset_grey=None, center_x=None, center_y=None,
                  length_x=None, length_y=None, resolution=1,
                  scalebar_width=None, scalebar_height=None, annotations=None,
                  **kwargs):
    image_width, image_height = get_image_shape(filenames[0])
    center_x = check_set(center_x, (image_width - 1) / 2)
    center_y = check_set(center_y, (image_height - 1) / 2)
    length_x = check_set(length_x, image_width)
    length_y = check_set(length_y, image_height)
    bin_by = check_set(bin_by, 1)
    bin_by = np.asarray(bin_by).flatten().tolist()
    if len(bin_by) == 1:
        bin_by.append(bin_by[0])

    # Calculate the binned dimensions needed for the 16 pixels of the mp4 codec
    multiple_of = [16]*2
    for i, bb in enumerate(bin_by):
        multiple_of[i] = multiple_of[i] * bb
    # Get the crop region of the image
    roi = get_crop_image_roi(image_width, image_height, center_x, center_y,
                             length_x, length_y, multiple_of=multiple_of)
    idx_x = slice(roi[0], roi[1])  # columns
    idx_y = slice(roi[2], roi[3])  # rows

    # Load and crop the background image
    if background_image is None:
        background = 0
    else:
        background = pims.open(background_image)[0][idx_y, idx_x]

    # Calculate the binned resolution in x and y
    resolution = check_set(resolution, 1.0)
    resolution = np.asarray(resolution).flatten().tolist()
    if len(resolution) == 1:
        resolution.append(resolution[0])
    for i, bb in enumerate(bin_by):
        resolution[i] = resolution[i] * bb

    annotations = check_set(annotations, [])
    # Calculate changes due to binning factor and protect original annotations
    # list
    _annotations = []
    for annotation in annotations:
        idcs = annotation[0]
        pos = np.asarray(annotation[1]).flatten().tolist()
        pos[0] = pos[0] // bin_by[0]
        pos[1] = pos[1] // bin_by[1]
        size = annotation[2] // bin_by[0]
        text = annotation[3]
        _annotations.append([idcs, pos, size, text])
    annotations = _annotations

    # pixelformat='gray16le'
    writer = imageio.get_writer(savename, **kwargs)

    for i, filename in enumerate(filenames):
        # Read, crop and subtract background from image
        im = pims.open(filename)[0][idx_y, idx_x] - background
        # Bin the image
        im = bin_image(im, bin_by=bin_by, func=np.mean)
        # Convert image into 8bit image and adjust the grey levels
        min_grey, max_grey = get_minmax_grey(im, min_grey, max_grey,
                                             width_grey, offset_grey)
        im = convert_image(im, 'uint8', min_grey, max_grey)
        # Draw a scalebar
        if scalebar_width is not None:
            scalebar(im, resolution, width=scalebar_width,
                     height=scalebar_height)
        for annotation in annotations:
            if i in annotation[0]:
                pos = np.asarray(annotation[1])
                size = annotation[2]
                text = annotation[3]
                im = annotate(im, pos, size, text)
        writer.append_data(im)

    writer.close()


def create_video(directory, prefix=None, suffix=None, extension=None,
                 sort_key=None, start_image_i=None, stop_image_i=None,
                 split_time=None, split_regexp=None, fps=None, fps_speedup=1.0,
                 decimate=1, background_image=None, bin_by=None, min_grey=None,
                 max_grey=None, width_grey=None, offset_grey=None,
                 center_x=None, center_y=None, length_x=None, length_y=None,
                 resolution=1, scalebar_width=None, scalebar_height=None,
                 annotations=None, videoname=None, videosuffix='.mp4',
                 videodirectory=None, **kwargs):
    """
    Parameters
    ----------
    directory : str
        The directory to look in for image files to create the video with.
    prefix : str
        A prefix to filter filenames with. Only filenames of the form "prefix*"
        will be used for video creation. Defaults to 'None'.
    suffix : str
        A suffix to filter filenames with. Only filenames of the form
        '`prefix`*`suffix``extension`' will be used for video creation.
        Defaults to 'None'.
    extension : str
        The extension the filenames have to have. Defaults to 'None'.
    sort_key : func
        A function to sort the filenames with. Defaults to `creation_time`.
    start_image_i : int
        The index of the first file from the sorted filenames to be used for
        the video.
    stop_image_i : int
        The stop index of the file to be used for the video. Only filenames up
        to index `stop_image_i` - 1 will be used.
    split_time : float
        The time in s, upon which two consecutive files/images are considered
        to belong to different videos and are therefore split into different
        video files.
    split_regexp : str
        Regular expression to extract the split_time in ms from the filenames.
        For instance, to extract the time from filenames like
        'B14-029ms-8902.tif', one can use the regexp: '-([0-9]*)ms-'.
    fps : float or str
        Autodetect or set the frames per second of the source files.
        'predominant' calculates the median of all creation time differences
        between all files and uses the reciprocal as fps. 'total' uses the
        difference of the creation time of the last and the first file and
        divides it by the total number of images. 'no_fps' sets the fps to
        'None'. This is usefull for videoformat (see `videosuffix`) not
        suporting setting the `fps`. Defaults to 'predominant'.
    fps_speedup : float
        How much to speed up the video from realtime.
    decimate : int
        How many images to jump from frame to frame.
    background_image = str
        Path to a background image file.
    bin_by : int or (int, int)
        Factor of how much pixels of the original image should be binned in the
        two dimensions (x, y). If only one int is given, assume x=y. Defaults
        to (1, 1).
    resolution : float
        Resolution of the image in units/px. This value is used to calculate
        the size of the scalebar.
    annotations : list of (indices, (x, y), size, text)
        A list of annotations. An annotation consists of an indexible 4tuple.
        'indices' are the indices of the images of the final movie, which
        should be annotated. This can be any object supporting the 'in'
        operator. '(x, y)' are the coordinates in pxs of the lower left corner
        of the text. 'size' is the size of the font. 'text' is the text used
        for the annotation.
    **kwargs : dict
        `kwargs` is passed to the function `imageio.get_writer`.
    """
    sort_key = check_set(sort_key, creation_time)
    filenames = files(directory, prefix=prefix, suffix=suffix,
                      extension=extension, sort_key=sort_key)
    filenames = filenames[start_image_i:stop_image_i]
    filelists = split_filenames(filenames, split_time=split_time,
                                regexp=split_regexp)

    videoname = check_set(videoname, prefix)
    videodirectory = check_set(videodirectory, os.path.join(directory, '..'))

    for i, (filenames, start, stop) in enumerate(filelists):
        # Create video, if the number of images is sufficient
        if len(filenames) >= 2:
            _filenames = filenames[::decimate]
            if len(filelists) == 1:
                _videoname = ''.join((videoname, videosuffix))
            else:
                _videoname = ''.join((videoname, '_{:02d}'.format(i+1),
                                      videosuffix))
            savename = os.path.join(videodirectory, _videoname)

            print('Creating Video \'{}\' of {} files ...'.format(
                                                    savename, len(_filenames)))
            print('  Frame slice {}:{}'.format(start, stop))
            fps_source = get_fps(filenames, fps=fps)
            if fps_source is not None:
                fps_target = fps_source * fps_speedup / decimate
                kwargs['fps'] = fps_target
                print('  Frames per second source: {:.2f}'.format(fps_source))
                print('  Frames per second target: {:.2f}'.format(fps_target))

            _create_video(_filenames, savename,
                          background_image=background_image, bin_by=bin_by,
                          min_grey=min_grey, max_grey=max_grey,
                          width_grey=width_grey, offset_grey=offset_grey,
                          center_x=center_x, center_y=center_y,
                          length_x=length_x, length_y=length_y,
                          resolution=resolution, scalebar_width=scalebar_width,
                          scalebar_height=scalebar_height,
                          annotations=annotations, **kwargs)
            print('  ... done.')


def check_framerate(filenames, tdiff_max, bins):
    # Plot the creation time of the files
    ct = np.array([creation_time(f) for f in filenames])

    fig, ax = plt.subplots()
    ax.plot(ct)

    # Show the histogram of the differences of creation times of the images
    tdiffs = ct[1:] - ct[:-1]
    print('Image files that have been overwritten: {:.1f} %'.format(
        len(tdiffs[tdiffs > tdiff_max]) / len(tdiffs) * 100))
    tdiffs_filtered = tdiffs[tdiffs < tdiff_max]
    print(
        'Number of time differences that are ignored in histogram: {}'.format(
            len(tdiffs) - len(tdiffs_filtered)))

    fig2, ax2 = plt.subplots()
    ax2.hist(tdiffs_filtered, bins=bins)
    ax2.set_yscale('log')
    ax2.set_xlim(tdiffs_filtered.min(), tdiffs_filtered.max())
    ax2.set_title('Time differences between recorded images from the Video.vi')
    ax2.set_xlabel('Image time difference (s)')
    ax2.set_ylabel('Number of occasions')

    return fig, ax, fig2, ax2
