#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import zipfile
import bz2
import io
import functools
import sys
import numpy as np
import pandas as pd
import collections
from scipy.sparse import coo_matrix
import hicstuff.hicstuff as hcs
from hicstuff.log import logger

DEFAULT_MAX_MATRIX_SHAPE = 10000

DEFAULT_FRAGMENTS_LIST_FILE_NAME = "fragments_list.txt"
DEFAULT_INFO_CONTIGS_FILE_NAME = "info_contigs.txt"
DEFAULT_SPARSE_MATRIX_FILE_NAME = "abs_fragments_contacts_weighted.txt"

load_raw_matrix = functools.partial(np.loadtxt, skiprows=1, delimiter="\t")


def raw_cols_to_sparse(M, dtype=np.float64):
    """
    Make a coordinate based sparse matrix from columns.
    Convert (3, n) shaped arrays to a sparse matrix. The fist
    one acts as the row coordinates, the second one as the
    column coordinates, the third one as the data points
    for each pair. If duplicates are found, the data points are
    added.

    Parameters
    ----------
    M : array_like
        An array with exactly three rows representing the sparse
        matrix in coordinate format.
    dtype : type, optional
        The type of data being loaded. Default is numpy.float64

    Example
    -------

        >>> import numpy as np
        >>> row, col = np.array([1, 2, 3]), np.array([3, 2, 1])
        >>> data = np.array([4, 5, 6])
        >>> M = np.array([row, col, data]).T
        >>> S = raw_cols_to_sparse(M)
        >>> print(S.todense())
        [[0. 0. 0. 0.]
         [0. 0. 0. 4.]
         [0. 0. 5. 0.]
         [0. 6. 0. 0.]]
    """
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
    """Save a sparse matrix

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
        path,
        delimiter="\t",
        usecols=(colnum,),
        skip_header=header,
        dtype=dtype,
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
            raise IOError(
                "Only a single fastq file must be in the zip archive."
            )
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
    for magic, _ in comp_bytes.items():
        if file_start.startswith(magic):
            return True
    return False


def from_dade_matrix(filename, header=False):
    """Load a DADE matrix

    Load a numpy array from a DADE matrix file, optionally
    returning bin information from the header. Header data
    processing is delegated downstream.

    Parameters
    ----------
    filename : str, file or pathlib.Path
        The name of the file containing the DADE matrix.
    header : bool
        Whether to return as well information contained
        in the header. Default is False.

    Example
    -------
        >>> import numpy as np
        >>> import tempfile
        >>> lines = [['RST', 'chr1~0', 'chr1~10', 'chr2~0', 'chr2~30'],
        ...          ['chr1~0', '5'],
        ...          ['chr1~10', '8', '3'],
        ...          ['chr2~0', '3', '5', '5'],
        ...          ['chr2~30', '5', '10', '11', '2']
        ...          ]
        >>> formatted = ["\\t".join(l) + "\\n" for l in lines ]
        >>> dade = tempfile.NamedTemporaryFile(mode='w')
        >>> for fm in formatted:
        ...     dade.write(fm)
        34
        9
        12
        13
        18
        >>> dade.flush()
        >>> M, h = from_dade_matrix(dade.name, header=True)
        >>> dade.close()
        >>> print(M)
        [[ 5.  8.  3.  5.]
         [ 8.  3.  5. 10.]
         [ 3.  5.  5. 11.]
         [ 5. 10. 11.  2.]]

        >>> print(h)
        ['chr1~0', 'chr1~10', 'chr2~0', 'chr2~30']

    See https://github.com/scovit/DADE for more details about Dade.
    """

    A = pd.read_csv(filename, sep="\t", header=None)
    A.fillna("0", inplace=True)
    M, headers = np.array(A.iloc[1:, 1:], dtype=np.float64), A.iloc[0, :]
    matrix = M + M.T - np.diag(np.diag(M))
    if header:
        return matrix, headers.tolist()[1:]
    else:
        return matrix


def to_dade_matrix(M, annotations="", filename=None):
    """Returns a Dade matrix from input numpy matrix. Any annotations are added
    as header. If filename is provided and valid, said matrix is also saved
    as text.
    """

    n, m = M.shape
    A = np.zeros((n + 1, m + 1))
    A[1:, 1:] = M
    if not annotations:
        annotations = np.array(["" for _ in n], dtype=str)
    A[0, :] = annotations
    A[:, 0] = annotations.T
    if filename:
        try:
            np.savetxt(filename, A, fmt="%i")
            logger.info(
                "I saved input matrix in dade format as " + str(filename)
            )
        except ValueError as e:
            logger.warning("I couldn't save input matrix.")
            logger.warning(str(e))

    return A


def load_into_redis(filename):
    """Load a file into redis

    Load a matrix file and sotre it in memory with redis.
    Useful to pass around huge datasets from scripts to
    scripts and load them only once.

    Inspired from https://gist.github.com/alexland/ce02d6ae5c8b63413843

    Parameters
    ----------
    filename : str, file or pathlib.Path
        The file of the matrix to load.

    Returns
    -------
    key : str
        The key of the dataset needed to retrieve it from redis.
    """
    try:
        from redis import StrictRedis as redis
        import time
    except ImportError:
        print(
            "Error! Redis does not appear to be installed in your system.",
            file=sys.stderr,
        )
        exit(1)

    M = np.genfromtxt(filename, dtype=None)
    array_dtype = str(M.dtype)
    m, n = M.shape
    M = M.ravel().tostring()
    database = redis(host="localhost", port=6379, db=0)
    key = "{0}|{1}#{2}#{3}".format(int(time.time()), array_dtype, m, n)

    database.set(key, M)

    return key


def load_from_redis(key):
    """Retrieve a dataset from redis

    Retrieve a cached dataset that was stored in redis
    with the input key.

    Parameters
    ----------
    key : str
        The key of the dataset that was stored in redis.
    Returns
    -------
    M : numpy.ndarray
        The retrieved dataset in array format.
    """

    try:
        from redis import StrictRedis as redis
    except ImportError:
        print(
            "Error! Redis does not appear to be installed in your system.",
            file=sys.stderr,
        )
        exit(1)

    database = redis(host="localhost", port=6379, db=0)

    try:
        M = database.get(key)
    except KeyError:
        print(
            "Error! No dataset was found with the supplied key.",
            file=sys.stderr,
        )
        exit(1)

    array_dtype, n, m = key.split("|")[1].split("#")

    M = np.fromstring(M, dtype=array_dtype).reshape(int(n), int(m))
    return M


def dade_to_GRAAL(
    filename,
    output_matrix=DEFAULT_SPARSE_MATRIX_FILE_NAME,
    output_contigs=DEFAULT_INFO_CONTIGS_FILE_NAME,
    output_frags=DEFAULT_SPARSE_MATRIX_FILE_NAME,
    output_dir=None,
):
    """Convert a matrix from DADE format (https://github.com/scovit/dade)
    to a GRAAL-compatible format. Since DADE matrices contain both fragment
    and contact information all files are generated at the same time.
    """
    import numpy as np

    with open(output_matrix, "w") as sparse_file:
        sparse_file.write("id_frag_a\tid_frag_b\tn_contact")
        with open(filename) as file_handle:
            first_line = file_handle.readline()
            for row_index, line in enumerate(file_handle):
                dense_row = np.array(line.split("\t")[1:], dtype=np.int32)
                for col_index in np.nonzero(dense_row)[0]:
                    line_to_write = "{}\t{}\t{}\n".format(
                        row_index, col_index, dense_row[col_index]
                    )
                    sparse_file.write(line_to_write)

    header = first_line.split("\t")
    bin_type = header[0]
    if bin_type == '"RST"':
        logger.info("I detected fragment-wise binning")
    elif bin_type == '"BIN"':
        logger.info("I detected fixed size binning")
    else:
        logger.warning(
            (
                "Sorry, I don't understand this matrix's "
                "binning: I read {}".format(str(bin_type))
            )
        )

    header_data = [
        header_elt.replace("'", "")
        .replace('"', "")
        .replace("\n", "")
        .split("~")
        for header_elt in header[1:]
    ]

    (
        global_frag_ids,
        contig_names,
        local_frag_ids,
        frag_starts,
        frag_ends,
    ) = np.array(list(zip(*header_data)))

    frag_starts = frag_starts.astype(np.int32) - 1
    frag_ends = frag_ends.astype(np.int32) - 1
    frag_lengths = frag_ends - frag_starts

    total_length = len(global_frag_ids)

    with open(output_contigs, "w") as info_contigs:

        info_contigs.write("contig\tlength\tn_frags\tcumul_length\n")

        cumul_length = 0

        for contig in collections.OrderedDict.fromkeys(contig_names):

            length_tig = np.sum(frag_lengths[contig_names == contig])
            n_frags = collections.Counter(contig_names)[contig]
            line_to_write = "%s\t%s\t%s\t%s\n" % (
                contig,
                length_tig,
                n_frags,
                cumul_length,
            )
            info_contigs.write(line_to_write)
            cumul_length += n_frags

    with open(output_frags, "w") as fragments_list:

        fragments_list.write(
            "id\tchrom\tstart_pos\tend_pos" "\tsize\tgc_content\n"
        )
        bogus_gc = 0.5

        for i in range(total_length):
            line_to_write = "%s\t%s\t%s\t%s\t%s\t%s\n" % (
                int(local_frag_ids[i]) + 1,
                contig_names[i],
                frag_starts[i],
                frag_ends[i],
                frag_lengths[i],
                bogus_gc,
            )
            fragments_list.write(line_to_write)


def load_bedgraph2d(filename):
    """
    Loads matrix and fragment information from a 2D bedgraph file.
    Parameters
    ----------
    filename : str
        Path to the bedgraph2D file.
    Returns
    -------
    mat : scipy.sparse.coo_matrix
        The Hi-C contact map as the upper triangle of a symetric matrix, in
        sparse format.
    frags : pandas.DataFrame
        The list of fragments/bin present in the matrix with their genomic
        positions.
    """
    bed2d = pd.read_csv(filename, sep=" ", header=None)
    # Get unique identifiers for fragments (chrom+pos)
    frag_pos_a = np.array(
        bed2d[[0, 1]].apply(lambda x: "".join(x.astype(str)), axis=1).tolist()
    )
    frag_pos_b = np.array(
        bed2d[[3, 4]].apply(lambda x: "".join(x.astype(str)), axis=1).tolist()
    )
    # Match position-based identifiers to their index
    ordered_frag_pos = np.unique(np.concatenate([frag_pos_a, frag_pos_b]))
    frag_map = {v: i for i, v in enumerate(ordered_frag_pos)}
    frag_id_a = np.array(list(map(lambda x: frag_map[x], frag_pos_a)))
    frag_id_b = np.array(list(map(lambda x: frag_map[x], frag_pos_b)))
    contacts = np.array(bed2d.iloc[:, 6].tolist())
    # Use index to build matrix
    n_frags = len(frag_map.keys())
    mat = coo_matrix(
        (contacts, (frag_id_a, frag_id_b)), shape=(n_frags, n_frags)
    )
    frags = (
        bed2d.groupby([0, 1], sort=False)
        .first()
        .reset_index()
        .iloc[:, [0, 1, 2]]
    )
    frags[3] = frags.iloc[:, 2] - frags.iloc[:, 1]
    frags.insert(loc=0, column="id", value=0)
    frags.columns = ["id", "chrom", "start_pos", "end_pos", "size"]
    return mat, frags
