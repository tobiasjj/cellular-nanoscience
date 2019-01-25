#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# cn_plot_style, set matplotlib according to the Cellular Nanosciencede group
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
Set matplotlib parameters to agree with the Cellular Nanoscience group plotting
conventions and offer a convenient ContextManager, to plot within a with
statement.

Examples
--------
import cn_plot_style as cnps
import numpy as np
from matplotlib import pyplot as plt


# Create data to be plotted
x = np.linspace(0, 50, 2000)
data = np.sin(x)
noisy_data = data + np.random.randn(2000) * 5


# Set plotting parameters for a 'notebook' with dark background:
cnps.set_plot_params(context='notebook', dark=True)

# Reset plotting parameters:
cnps.reset_plot_params()

# Temporarily set plotting parameters for a 'notebook' with dark background and
# plot data on one axis, and cycle through colors:
with cnps.cn_plot(context='notebook', dark=True):
    fig, ax = plt.subplots()
    lns1 = ax.plot(x, noisy_data, label='data')
    lns2 = ax.plot(x, data, label='fit')

    cnps.legend(lns1, lns2)

    ax.set_title('Example fit of a sinus')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

# Plot data for a paper with 'twocolumn's on one axis, and cycle through colors
# and dashes:
with cnps.cn_plot(context='twocolumn', dash=True):
    fig, ax = plt.subplots()
    lns1 = ax.plot(x, noisy_data, label='data')
    lns2 = ax.plot(x, data, label='fit')

    cnps.legend(lns1, lns2)

    ax.set_title('Example fit of a sinus')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

# Plot data for a paper with a width of 85 mm, on one axis, black and white,
# and cycle through dashes:
with cnps.cn_plot(context='paper', fig_width=85, unit='mm', color=False,
                  color_index=0, dash=True):
    fig, ax = plt.subplots()
    lns1 = ax.plot(x, noisy_data, label='data')
    lns2 = ax.plot(x, data, label='fit')

    cnps.legend(lns1, lns2)

    ax.set_title('Example fit of a sinus')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

# Plot data for a 'paper' on two axes, sharing the x axis, with different y
ticks, and cycle through colors and dashes:
from mpl_toolkits.axes_grid1 import host_subplot, host_axes
with cnps.cn_plot(context='paper', right_spine=True, dash=True) as cnp:
    ax = host_subplot(111)
    ax2 = ax.twinx()

    lns1 = ax.plot(x, noisy_data, label='data')
    cnp.set_axis_color(ax=ax)

    lns2 = ax2.plot(x, data, label='fit')
    cnp.set_axis_color()

    cnps.legend(lns1, lns2)

    ax.set_title('Example fit of a sinus')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax2.set_ylabel('Amplitude')

# Plot data for a 'paper' with two independent axes, and cycle through colors
# and dashes, with a figure width of 373.44 pt and an aspect_ratio of 4/3.
# Finally, save the figure (the save command should be within the with
# statement).
with cnps.cn_plot(context='paper', dash=True, fig_width=373.44, unit='pt',
                  aspect_ratio=4/3) as cnp:
    fig, ax = plt.subplots()
    ax2 = cnps.second_ax(fig=fig, link_ax=ax)
    # fig.add_subplot(111, frame_on=False)
    # ax2.xaxis.set_label_position('top')
    # ax2.yaxis.set_label_position('right')
    # ax2.xaxis.tick_top()
    # ax2.yaxis.tick_right()
    # cnps.link_ax_cycle(ax, ax2)

    lns1 = ax.plot(x, noisy_data, label='data')
    cnp.set_axis_color(ax=ax)

    lns2 = ax2.plot(x, data, label='fit')
    cnp.set_axis_color(ax=ax2)

    # cnps.legend(lns1, lns2)

    ax.set_xlabel(r'Time (a.\,u.)')
    ax.set_ylabel(r'Amplitude (\textmu m)')
    ax2.set_xlabel(r'Time2 (a.\,u.)')
    ax2.set_ylabel(r'Light sheet length (\textmu m)')

    plt.savefig('test.png')
"""
__author__ = "Tobias Jachowski"
__copyright__ = "Copyright 2017"
__credits__ = "Moritz Burmeister"
__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Tobias Jachowski"
__email__ = "cn_plot_style@jachowski.de"
__status__ = "stable"

from scipy import constants
from cycler import cycler
from matplotlib import pyplot as plt
import itertools
import operator


_light_colors = [
    # blue, orange, green, violet, brown, magenta, grey, limish, cyan, red
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']

_dark_colors = [
    '#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2',
    '#fdb462', '#b3de69', '#bc82bd', '#ccebc4', '#ffed6f']

_light_greys = [
    '0.0', '0.3', '0.45', '0.6', '0.7',
    '0.0', '0.3', '0.45', '0.6', '0.7']

_dark_greys = [
    '1.0', '0.8', '0.6', '0.45', '0.3',
    '1.0', '0.8', '0.6', '0.45', '0.3']

_markers = [*'oDshv^<>pP']


_dashes = [
        (),                        # line
        (3, 1),                    # dash
        (5, 1, 1, 1),              # daash, dot
        (5, 2),                    # daash
        (5, 1, 1, 1, 1, 1),        # daash, dot, dot
        (1, 1),                    # dots
        (5, 1, 5, 1, 5, 3),        # dash, dash, dash
        (10, 1, 1, 1, 1, 2),       # daaaash, dot, dot
        (10, 2, 5, 2),             # daaaash, daash
        (10, 2, 1, 2, 5, 2, 1, 2)  # daaaash, dot, daash, dot
    ]

# Figure width according to context (default to 6.89 inch, i.e. a typical
# textwidth for a paper)
_context_width = {
    'default': 6.89,
    'twocolumn': 3.35,
    'notebook': 5,
    'talk': 5.17,
    'poster': 10
}

# Figure aspect ratio according to the context (default to the golden ratio)
_context_aspect = {
    'default': constants.golden_ratio,
    'notebook': 4 / 3,
    'talk': 4 / 3
}

# Scaling of line thickness and font size according to the context (default
# scaling appropriate for a paper, i.e. 1.0)
_context_scale = {
    'default': 1.0,
    'notebook': 1.15,
    'talk': 1.5,
    'poster': 2.0
}

_orig_rcParams = None


def linewidth_typearea(papersize=None, font_size=None, DIV=None, BCOR=None):
    """
    Calculate the width according to parameters used by typearea in a LaTeX
    document.

    papersize : str
        Right now, only 'a4' is supported.
    font_size : str
        The font size is used to automatically determine `DIV`. Right now,
        following values are supported: '10pt', '11pt', or '12pt'. Defaults to
        '10pt'.
    DIV : int
        The number of parts the page of a paper is divided into. Defaults to 8.
    BCOR : float
        Binding correction in mm.

    Returns
    -------
    float
        The width in mm.
    """
    DIV_font_size = {
        '10pt': 8,
        '11pt': 10,
        '12pt': 12
    }

    paperwidth = {
        'a4': 210.0
    }

    width = paperwidth.get(papersize, 210.0)
    DIV = DIV or DIV_font_size.get(font_size, 8)
    BCOR = BCOR or 0.0
    width_mm = (width - BCOR)/DIV * (DIV - 3.0)

    return width_mm


def convert_length(length=1.0, from_unit='pt', to_unit='in'):
    """
    Convert the length in point of an object in a LaTeX document into another
    unit.

    For different point definitions see:
    https://tex.stackexchange.com/questions/200934/why-does-a-tex-point-differ-
    from-a-desktop-publishing-point#200968

    Parameters
    ----------
    length : float
        Length of an object in a LaTeX document. For instance, the width of
        text can be determined by using \showthe\linewidth or
        \showthe\columnwidth. See https://en.wikibooks.org/wiki/LaTeX/Lengths.
    from_unit : str
        pt, pc, bp, mm, or in
        A tex point is defined as 1/72.27 inch!
        pt pica point
        pc Pica
        bp big point, rounded point, PostScript point
    to_unit : str
        pt, pc, bp, mm, or in

    Returns
    -------
    float
        The length in unit `to_unit`.
    """
    # From and to conversion factors
    pt_per = {
        'pt': 1.0,
        'pc': 12.0,
        'bp': 72.27 / 72,
        'mm': 72.27 / 25.4,
        'in': 72.27
    }
    pt_to = {
        'pt': 1.0,
        'pc': 1.0 / 12.0,
        'bp': 72 / 72.27,
        'mm': 25.4 / 72.27,
        'in': 1 / 72.27
    }

    length = length * pt_per[from_unit] * pt_to[to_unit]

    return length


def fig_size(context='default', fig_width=None, unit='in', aspect_ratio=None,
             fig_width_scale=1.0, fig_height_scale=1.0, fig_scale=1.0,
             **kwargs):
    """
    Parameters
    ----------
    context : str
        The context for which the figure size should be calculated.
    fig_width : float
        The width of the figure in inch. Defaults to a predefined width which
        is reasonable for a given `context`.
    aspect_ratio : float
        Ratio of width to height of the figure. Depending on the `context` the
        ratio defaults to the aestethic ratio golden ratio ('paper', 'poster',
        and 'notebook') or 4:3 ('talk'). The aspect ratio will be modified by
        setting `fig_width_scale` and/or `fig_height_scale`.
    fig_width_scale : float
        Scaling factor of the figure width.
    fig_height_scale : float
        Scaling factor of the figure height.
    fig_scale: float
        Overall scaling factor of the figure size.
    **kwargs : dict
        Not used.

    Returns
    -------
    tuple of 2 floats
        Figure size (width,  height).
    """
    if fig_width is None:
        # Get the width according to context
        default = _context_width['default']
        fig_width = _context_width.get(context, default)
    else:
        # Set and convert unit of fig_width to inch
        fig_width = convert_length(length=fig_width, from_unit=unit)

    # Get the aspect ratio
    default = _context_aspect['default']
    aspect_ratio = aspect_ratio or _context_aspect.get(context, default)

    fig_width = fig_width * fig_scale
    fig_height = fig_width / aspect_ratio * fig_height_scale

    fig_width = fig_width * fig_width_scale

    return fig_width, fig_height


def _colors(dark=False, color=True, color_index=None, **kwargs):
    """
    Determine the colors to be used.
    """
    if color_index is None:
        idx = slice(None)
    else:
        idx = slice(color_index, color_index + 1)

    if color:
        if dark:
            return _dark_colors.copy()[idx]
        else:
            return _light_colors.copy()[idx]
    else:
        if dark:
            return _dark_greys.copy()[idx]
        else:
            return _light_greys.copy()[idx]


def cyclers(color=True, marker=False, dash=False, colors=None, markers=None,
            dashes=None, dark=False, color_index=None, **kwargs):
    """
    Create a cycler object which cycles over colors, markers and dashes.
    """
    markers = markers or _markers
    dashes = dashes or _dashes

    colors = colors or _colors(dark=dark, color=color, color_index=color_index)
    cl = cycler(color=colors)

    if color_index is None:
        op = operator.add
    else:
        op = operator.mul

    if dash:
        cl = op(cl, cycler(dashes=dashes))
        op = operator.add

    if marker:
        cl = op(cl, cycler(marker=markers))

    return cl


def theme(dark=False, lighten=0.3, lighten_edges=None, lighten_text=None,
          lighten_grid=None, **kwargs):
    """
    Create plotting parameters for different themes.

    Parameters
    ----------
    dark : bool
        Dark or light theme.
    lighten : float with range [0.0, 1.0]
        Lighten lines by fration `lighten`.
    lighten_edges : float
        Defaults to `lighten`.
    lighten_text : float
        Defaults to `lighten`.
    lighten_grid : float
        Defaults to `lighten`.
    **kwargs : dict
        Not used.

    Returns
    -------
    dict
        Matplotlib plotting parameters dictionary.
    """
    if lighten_edges is None:
        lighten_edges = lighten
    if lighten_text is None:
        lighten_text = lighten
    if lighten_grid is None:
        lighten_grid = lighten_edges
    foreground_text = str(0.0 + lighten_text)
    foreground_edges = str(0.0 + lighten_edges)
    foreground_grid = str(0.0 + lighten_grid)
    background = 'white'
    if dark:
        foreground_text = str(1.0 - lighten_text)
        foreground_edges = str(1.0 - lighten_edges)
        foreground_grid = str(1.0 - lighten_grid)
        background = 'black'

    params = {
        # 'lines.color': foreground,
        'patch.edgecolor': foreground_edges,
        'text.color': foreground_text,
        'axes.facecolor': background,
        'axes.edgecolor': foreground_edges,
        'axes.labelcolor': foreground_text,
        'xtick.color': foreground_text,
        'ytick.color': foreground_text,
        'grid.color': foreground_grid,
        'legend.edgecolor': foreground_edges,
        'figure.facecolor': background,
        'figure.edgecolor': background,
        'savefig.facecolor': background,
        'savefig.edgecolor': background
    }

    return params


def plot_params(context='default', figsize=None, unit='in', scale=1.0,
                context_scale=None, data_scale=1.0, edges_scale=1.0,
                line_scale=1.0, marker_scale=1.0, pad_scale=1.0,
                text_scale=1.0, tick_scale=1.0, cycle=None, usetex=False,
                right_spine=False, top_spine=False, right_ticks=False,
                top_ticks=False, autolayout=True, fig_dpi=150, save_dpi=300,
                transparent_save=True, latex_preamble=None, **kwargs):
    """
    Create plotting parameters with the plotting style of the Cellular
    Nanoscience group.

    ######### Cookbook/Matplotlib/LaTeX Examples ###########
    ##### Producing Graphs for Publication using LaTeX #####
    text_scale gibt die Größenänderung des Textes in Bezug zur normalen Größe
    an.
      Eine Auflösung von 600 dpi sollte für die meisten Drucker ausreichend
    sein (dpi=600). Angegeben wird die gewünschte Auflösung des Plots im
    fertigen LaTeX Dokument. Sollte ein pdf Dokuement erstellt werde, anstelle
    einer png Grafik, hat die dpi Einstellung keine Auswirkungen.

    # Auflösung des Bildes für Drucker auf 300dpi stelle.
    # Falls pdf als backend verwendet wird, ist dpi Angabe unwichtig.
    """
    if figsize is None:
        # Determine the figsize according to the context and the overall
        # scaling factor
        fig_scale = kwargs.pop('fig_scale', 1.0) * scale
        figsize = fig_size(context=context, unit=unit, fig_scale=fig_scale,
                           **kwargs)
    else:
        # Convert units of figsize to inch
        width = convert_length(length=figsize[0], from_unit=unit)
        height = convert_length(length=figsize[1], from_unit=unit)
        figsize = (width, height)

    # Determine overall scaling according to context
    default = _context_scale['default']
    context_scale = context_scale or _context_scale.get(context, default)

    # Get the color (marker/dashes) cycler for the lines
    cycle = cycle or cyclers(**kwargs)

    # Get the theme for the lines (dark/white, light edges/lines)
    params = theme(**kwargs)

    # either set the latex preamble to default or use user chosen packages
    latex_preamble = (latex_preamble
                      or r'\usepackage{upgreek},\usepackage[cmbright]{sfmath}')

    params.update({
        # Figure size
        'figure.figsize': figsize,

        # Cyclers
        'axes.prop_cycle': cycle,

        # Widths of lines
        'grid.linewidth': 0.25 * edges_scale * context_scale * scale,
        'lines.linewidth': 1.0 * (line_scale * data_scale * context_scale
                                  * scale),
        'patch.linewidth': 0.25 * edges_scale * context_scale * scale,
        'axes.linewidth': 0.5 * edges_scale * context_scale * scale,

        # Remove frame of plots
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': top_spine,
        'axes.spines.right': right_spine,

        # Size of markers
        'lines.markersize': 6 * (marker_scale * data_scale * context_scale
                                 * scale),
        'lines.markeredgewidth': 0.0 * (marker_scale * data_scale
                                        * context_scale * scale),

        # Fonts
        'text.usetex': usetex,
        #'text.latex.unicode': True,
        'text.latex.preamble': latex_preamble,
        'font.family': 'sans-serif',
        # 'font.sans-serif': ['Helvetica', 'FreeSans'],
        # 'mathtext.fontset': 'custom',
        'figure.titlesize': 13 * text_scale * context_scale * scale,
        'font.size': 9 * text_scale * context_scale * scale,
        'axes.titlesize': 12 * text_scale * context_scale * scale,
        'axes.labelsize': 9 * text_scale * context_scale * scale,
        'xtick.labelsize': 9 * text_scale * context_scale * scale,
        'ytick.labelsize': 9 * text_scale * context_scale * scale,
        'legend.fontsize': 9 * text_scale * context_scale * scale,

        # Xticks
        'xtick.top': top_ticks,
        'xtick.major.size': 3.2 * tick_scale * context_scale * scale,
        'xtick.minor.size': 1.8 * tick_scale * context_scale * scale,
        'xtick.major.width': 0.5 * tick_scale * context_scale * scale,
        'xtick.minor.width': 0.5 * tick_scale * context_scale * scale,

        # Yticks
        'ytick.right': right_ticks,
        'ytick.major.size': 3.1 * tick_scale * context_scale * scale,
        'ytick.minor.size': 1.7 * tick_scale * context_scale * scale,
        'ytick.major.width': 0.5 * tick_scale * context_scale * scale,
        'ytick.minor.width': 0.5 * tick_scale * context_scale * scale,

        # Positioning of text
        'xtick.major.pad': 2.8 * pad_scale * context_scale * scale,
        'ytick.major.pad': 2.8 * pad_scale * context_scale * scale,

        # Legend
        #'legend.framealpha': 0.0,
        #'legend.fancybox': False,
        'legend.numpoints': 1,
        'legend.handlelength': 3.0 * context_scale * scale,

        # Automatically adjust subplot parameters to make the plot fit the
        # figure
        'figure.autolayout': autolayout,

        # Resolution
        'figure.dpi': fig_dpi,
        'savefig.dpi': save_dpi,

        # Transparency of background for nbagg backend and for saved figures
        #'nbagg.transparent': False,
        'savefig.transparent': transparent_save
    })

    return params


def set_plot_params(*args, **kwargs):
    """
    Set matplotlib plotting parameters according to the Cellular Nanoscience
    plotting style.
    """
    global _orig_rcParams
    if _orig_rcParams is None:
        _orig_rcParams = plt.rcParams.copy()
    else:
        print('Original plotting parameters already stored.')
        print('Only setting new paramaters.')
    params = plot_params(*args, **kwargs)
    plt.rcParams.update(params)
    return params

def reset_plot_params():
    """
    Reset matplotlib plotting parameters.
    """
    global _orig_rcParams
    if _orig_rcParams is None:
        return
    plt.rcParams.clear()
    plt.rcParams.update(_orig_rcParams)
    _orig_rcParams = None

class cn_plot(object):
    """
    A Cellular Nanoscience plotting environemnt.

    `cn_plot` is used within a with statement, wherein it adjusts the
    matplotlib plotting parameters according to the Cellular Nanoscience style.
    The plotting parameters outside of the with statement are not affected. On
    top of adjusting the plotting parameters, `cn_plot` provides functions, to
    easily access the current value of cycled colors, dashes, and markers.
    """
    def __init__(self, *args, dark=False, color=True, color_index=None,
                 lighten=0.3, **kwargs):
        # Store matplotlib plotting parameters to be able to restore them later
        self._rcparams = plt.rcParams.copy()

        # Set default values for color, dashes, and markers and create
        # corresponding iterators
        self.bw = not color and color_index is None
        self.colors = _colors(dark=dark, color=color, color_index=color_index)
        self.dashes = _dashes.copy()
        self.markers = _markers.copy()
        self.iter_colors = iter(self.colors)
        self.iter_dashes = iter(self.dashes)
        self.iter_markers = iter(self.markers)
        self.lighten = lighten

        # Write given keyword arguments into general **kwargs dict and get
        # plotting parameters to be used
        kwargs['dark'] = dark
        kwargs['color'] = color
        kwargs['color_index'] = color_index
        kwargs['lighten'] = lighten
        self.rcdict = plot_params(*args, **kwargs)

        # Update matplotlib plotting parameters, if error occurs, reset to
        # original plotting parameters
        try:
            plt.rcParams.update(self.rcdict)
        except:
            plt.rcParams.clear()
            plt.rcParams.update(self._rcparams)
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Restore original plotting parameters
        plt.rcParams.clear()
        rcparams = self._rcparams.copy()
        # workaround to suppress deprecation warning for matplotlib >= 3.0
        rcparams.pop('examples.directory', None)
        rcparams.pop('text.latex.unicode', None)
        plt.rcParams.update(rcparams)
        # Workaround for text rendered with TeX
        if self.rcdict['text.usetex']:
            for key in ['text.usetex', #  'text.latex.unicode',
                        'text.latex.preamble']:
                plt.rcParams[key] = self.rcdict[key]

    def set_axis_color(self, last_line_color=True, color=None,
                       color_index=None, **kwargs):
        if not last_line_color:
            if color_index is None:
                color = color or self.color
            else:
                color = color or self.colors[color_index]

        lighten = self.lighten
        if self.bw:
            # The lightening encodes the grey value. Do no extra lightening.
            lighten = 0.0

        return set_axis_color(color=color, lighten=lighten, **kwargs)

    @property
    def color(self):
        color = next(self.iter_colors, None)
        if color is None:
            self.iter_colors = iter(self.colors)
            color = next(self.iter_colors)
        return color

    @property
    def dash(self):
        dash = next(self.iter_dashes, None)
        if dash is None:
            self.iter_dashes = iter(self.dashes)
            dash = next(self.iter_dashes)
        return dash

    @property
    def marker(self):
        marker = next(self.iter_markers, None)
        if marker is None:
            self.iter_markers = iter(self.markers)
            marker = next(self.iter_markers)
        return marker


def set_axis_color(ax=None, color=None, lighten=0.3, label=True, ticks=True,
                   tick_lines=True, tick_labels=True, axis='y'):
    ax = ax or plt.gca()

    color = color or ax.lines[-1].get_color()

    alpha = 1.0 - lighten

    axes_axis = []
    axes_ticklines = []
    axes_ticklabels = []
    if axis in ['x', 'X', 'both']:
        axes_axis.append(ax.xaxis)
        axes_ticklines.append(ax.get_xticklines())
        axes_ticklabels.append(ax.get_xticklabels())
    if axis in ['y', 'Y', 'both']:
        axes_axis.append(ax.yaxis)
        axes_ticklines.append(ax.get_yticklines())
        axes_ticklabels.append(ax.get_yticklabels())

    for ax_axis in axes_axis:
        if label:
            ax_axis.label.set_color(color)
            ax_axis.label.set_alpha(alpha)

    # ax.tick_params('y', colors=colors)
    if ticks and tick_lines:
        for ax_ticklines in axes_ticklines:
            for tick in ax_ticklines:
                tick.set_color(color)
                tick.set_alpha(alpha)
    if ticks and tick_labels:
        for ax_ticklabels in axes_ticklabels:
            for label in ax_ticklabels:
                label.set_color(color)
                label.set_alpha(alpha)

    return color


def legend(*lines, axis=None):
    """
    Create legends for given lines. This function also works for lines from
    different axes (e.g. lines plotted on a second axis, see `second_ax()`).

    Parameters
    ----------
    lines : Iterable of matplotlib Lines
        Lines, which should be included in the legend.
    axis : Matplotlib Axis
        The axis where the legend should be created.
    """
    ax = axis or plt.gca()
    lns = list(itertools.chain(*lines))
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)


def link_ax_cycle(ax1, ax2):
    """
    Link the property cyclers of two matplotib axes.

    Parameters
    ----------
    ax1 : Matplotlib Axis
        The axis whose lines property cycler is taken to set the one  of `ax2`.
    ax2 : Matplotlib Axis
        The axis whose lines property cycler is overwritten by the one from
        `ax1`.
    """
    ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler


def second_ax(fig=None, subplot_pos=None, link_ax=None, enable_spines=True,
              spines_ax=None):
    """
    Function to create a second axis.

    Parameters
    ----------
    fig : Matplotlib Figure
        Defaults to `plt.gcf()`.
    subplot_pos : tuple
        Defaults to (1, 1, 1).
    link_ax : Matplotlib Axis
        An axis the new axis' prop_cycler should be taken from (linked to). New
        lines plotted on either axis will follow the common order of color,
        dashes, and markers.
    enable_spines : bool
        Enable top and right spines (i.e. top and right border) of axis
        `spines_ax`.
    spines_ax : Matplotlib Axis
        Axis whose top and right spines should be made visible. Defaults to
        `fig.gca()`. Only evaluated, if `enable_spines` is True.
    """
    fig = fig or plt.gcf()
    subplot_pos = subplot_pos or (1, 1, 1)

    if enable_spines:
        spines_ax = spines_ax or fig.gca()
        spines_ax.spines['top'].set_visible(True)
        spines_ax.spines['right'].set_visible(True)

    ax = fig.add_subplot(*subplot_pos, frame_on=False)
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()

    if link_ax:
        link_ax_cycle(link_ax, ax)

    return ax
