#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import zipfile
import bz2
import io
import functools
import numpy as np
from scipy.sparse import coo_matrix
import hicstuff.hicstuff as hcs

DEFAULT_MAX_MATRIX_SHAPE = 10000

load_raw_matrix = functools.partial(np.genfromtxt, skip_header=True, dtype=np.float64)


def raw_cols_to_sparse(M, dtype=np.float64):
    n = int(np.amax(M[:, :-1]) + 1)

    row = M[:, 0]
    col = M[:, 1]
    data = M[:, 2]
    S = coo_matrix((data, (row, col)), shape=(n, n), dtype=dtype)
    return S


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
    B = hcs.bin_sparse(S, subsampling_factor=subsampling_factor)
    return B


def save_sparse_matrix(M, path):
    """
    Saves a sparse matrix object into tsv format.
    Parameters
    ----------
    M : scipy.sparse.coo_matrix
        The sparse matrix to save on disk
    path : str
        File path where the matrix will be stored
    """
    S_arr = np.vstack([M.row, M.col, M.data]).T

    np.savetxt(
        path,
        S_arr,
        header="id_fragment_a\tid_fragment_b\tn_contact",
        comments="",
        fmt="%i",
        delimiter="\t",
    )


def load_pos_col(path, colnum, header=1, dtype=np.int64):
    """
    Loads a single column of a TSV file with header into a numpy array.
    Parameters
    ----------
    path : str
        The path of the TSV file to load.
    colnum : int
        The 0-based index of the column to load.
    header : int
        Number of line to skip. By default the header is a single line.
    Returns
    -------
    numpy.array :
        A 1D numpy array with the
    """
    pos_arr = np.genfromtxt(
        path, delimiter="\t", usecols=(colnum,), skip_header=header, dtype=dtype
    )
    return pos_arr


def read_compressed(filename):
    """Read compressed file

    Opens the file in read mode with appropriate decompression algorithm.

    Parameters
    ----------
    filename : str
        The path to the input file
    Returns
    -------
    file-like object
        The handle to access the input file's content
    """

    # Standard header bytes for diff compression formats
    comp_bytes = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
        b"\x50\x4b\x03\x04": "zip",
    }

    max_len = max(len(x) for x in comp_bytes)

    def file_type(filename):
        """Guess file type

        Compare header bytes with those in the file and return type.
        """
        with open(filename, "rb") as f:
            file_start = f.read(max_len)
        for magic, filetype in comp_bytes.items():
            if file_start.startswith(magic):
                return filetype
        return "uncompressed"

    # Open file with appropriate function
    comp = file_type(filename)
    if comp == "gz":
        return gzip.open(filename, "rt")
    elif comp == "bz2":
        return bz2.BZ2File(filename, "rt")
    elif comp == "zip":
        zip_arch = zipfile.ZipFile(filename, "r")
        if len(zip_arch.namelist()) > 1:
            raise IOError("Only a single fastq file must be in the zip archive.")
        else:
            # ZipFile opens as bytes by default, using io to read as text
            zip_content = zip_arch.open(zip_arch.namelist()[0], "r")
            return io.TextIOWrapper(zip_content)
    else:
        return open(filename, "r")


def is_compressed(filename):
    """Check compression status

    Check if the input file is compressed from the first bytes.

    Parameters
    ----------
    filename : str
        The path to the input file

    Returns
    -------
    bool
        True if the file is compressed, False otherwise.
    """

    # Standard header bytes for diff compression formats
    comp_bytes = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
        b"\x50\x4b\x03\x04": "zip",
    }
    max_len = max(len(x) for x in comp_bytes)
    with open(filename, "rb") as f:
        file_start = f.read(max_len)
    for magic, filetype in comp_bytes.items():
        if file_start.startswith(magic):
            return True
    return False
