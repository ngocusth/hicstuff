#!/usr/bin/env python3
# coding: utf-8

"""Hi-C visualization

A lightweight library for quickly parsing, loading and
viewing contact maps in instaGRAAL or csv format.
"""


import numpy as np
import functools
from matplotlib import pyplot as plt
from scipy import sparse
from hicstuff.hicstuff import bin_sparse, bin_bp_sparse

SEABORN = False

try:
    import seaborn as sns

    SEABORN = True
except ImportError:
    pass

try:
    import hicstuff.hicstuff as hcs
except ImportError:
    print("Warning, hicstuff was not found - normalizations won't work")

DEFAULT_DPI = 500
DEFAULT_SATURATION_THRESHOLD = 99
DEFAULT_MAX_MATRIX_SHAPE = 10000

load_raw_matrix = functools.partial(
    np.genfromtxt, skip_header=True, dtype=np.float64
)


def raw_cols_to_sparse(M, dtype=np.float64):
    n = int(np.amax(M[:, :-1]) + 1)

    row = M[:, 0]
    col = M[:, 1]
    data = M[:, 2]
    S = sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=dtype)
    return S


def sparse_to_dense(M):

    D = M.todense()
    E = D + np.transpose(D) - 2 * np.diag(np.diag(D))
    return E


def plot_matrix(
    array, filename=None, vmin=0, vmax=None, dpi=DEFAULT_DPI, cmap="Reds"
):
    """A function that performs all the tedious matplotlib
    magic to draw a 2D array with as few parameters and
    as little whitespace as possible.

    Adjusted from https://github.com/koszullab/metaTOR
    """

    if vmax is None:
        vmax = np.percentile(array, DEFAULT_SATURATION_THRESHOLD)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.figure()
    if SEABORN:
        sns.heatmap(array, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        plt.imshow(
            array, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none"
        )
        plt.colorbar()
    plt.axis("off")
    if filename:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.0, dpi=dpi)
        del filename
    else:
        plt.show()


def load_sparse_matrix(M, binning=1):
    """Load sparse matrix

    Load a text file matrix into a sparse matrix object.

    Parameters
    ----------
    M : file, str or pathlib.Path
        The input matrix file in instaGRAAL format.
    binning : int or "auto"
        The binning to perform. If "auto", binning will
        be automatically inferred so that the matrix size
        will not go beyond (10000, 10000) in shape. That
        can be changed by modifying the DEFAULT_MAX_MATRIX_SHAPE
        value. Default is 1, i.e. no binning is performed

    Returns
    -------
    N : scipy.sparse.coo_matrix
        The output (sparse) matrix in COOrdinate format.
    """

    R = load_raw_matrix(M)
    S = raw_cols_to_sparse(R)
    if binning == "auto":
        n = max(S.shape) + 1
        subsampling_factor = n // DEFAULT_MAX_MATRIX_SHAPE
    else:
        subsampling_factor = binning
    B = bin_sparse(S, subsampling_factor=subsampling_factor)
    return B


def normalize(M, norm="SCN"):
    """Attempt to normalize if hicstuff is found, does nothing otherwise.
    """
    try:
        return hcs.normalize_sparse(M, norm=norm)
    except NameError:
        return M
