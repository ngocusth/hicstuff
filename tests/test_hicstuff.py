#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hicstuff testing

Basic tests for functions in the hicstuff library.
"""

import random
import numpy as np
import pytest
import hicstuff.hicstuff as hcs
from scipy.sparse import coo_matrix, csr_matrix
from inspect import signature, getmembers, isfunction

SIZE_PARAMETERS = ("matrix_size", [5, 10, 20, 50, 100])


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_scn(matrix_size):
    """Test SCN normalization

    Check whether a SCN-normalized matrix has all vectors
    summing to one. Tests both the sparse and dense algorithms.
    """
    M_d = np.random.random((matrix_size, matrix_size))
    M_s = coo_matrix(M_d)
    M_d += M_d.T
    N_d = hcs.normalize_dense(M_d, "SCN", iterations=50)
    assert np.isclose(N_d.sum(axis=1), np.ones(matrix_size), rtol=0.0001).all()
    N_s = hcs.normalize_sparse(M_s, "SCN", iterations=50)
    print(N_s.sum(axis=1))
    print(N_s.shape)
    print(N_s.nonzero())
    assert np.isclose(N_s.sum(axis=1), np.ones(matrix_size), rtol=0.0001).all()


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_trim(matrix_size):
    """
    Generate a random matrix and introduce outlier bins. Check if the correct
    number of bins are trimmed.
    """

    M_d = np.random.random((matrix_size, matrix_size))
    M_d += M_d.T

    def bin_stats(bins, n_std=3):
        # Get thresholds for trimming
        mean = np.mean(bins)
        sd = np.std(bins)
        min_val, max_val = (mean - n_std * sd, mean + n_std * sd)
        return min_val, max_val

    means = M_d.mean(axis=1)
    min_val, max_val = bin_stats(means)
    # Choose random bins
    trim_bins = np.random.randint(0, M_d.shape[0], 2)
    # Add potential pre-existing outlier bins
    trim_bins = np.append(
        trim_bins, np.where((means <= min_val) | (means >= max_val))[0]
    )
    # Set bins to outlier values
    M_d[:, trim_bins] = M_d[trim_bins, :] = random.choice([min_val, max_val])
    # Compute trimming thresholds again
    means = M_d.mean(axis=1)
    min_val, max_val = bin_stats(means)
    # Define bins that need to be trimmed
    trim_bins = np.where((means <= min_val) | (means >= max_val))[0]
    print(trim_bins)
    trim_shape = M_d.shape[0] - len(trim_bins)
    # Compare expected shape with results
    M_s = coo_matrix(M_d)
    T_d = hcs.trim_dense(M_d)
    assert T_d.shape[0] == trim_shape
    T_s = hcs.trim_sparse(M_s)
    assert T_s.shape[0] == trim_shape


@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_basic_one_argument_functions(matrix_size):
    """Check all functions

    Generate an NxN matrix at random and feed it to all functions
    with only one argument. This is meant to catch very fundamental
    errors and also facilitate runtime type guessing with MonkeyType.

    Parameters
    ----------
    matrix_size : int
        The size of the random matrix to use for the tests.
    """
    functions_list = getmembers(hcs, isfunction)
    M_d = np.random.random((matrix_size, matrix_size))
    M_d += M_d.T
    for _, func in functions_list:
        params = signature(func).parameters
        if func.__defaults__ is None:
            nb_defaults = 0
        else:
            nb_defaults = len(func.__defaults__)
        annot = params[list(params.keys())[0]].annotation

        if len(params) == 1 or len(params) - nb_defaults == 1:
            try:
                assert func(M_d).any()
            except (ValueError, TypeError, AttributeError):
                pass


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_corrcoef_sparse(matrix_size):
    """
    Checks if the corrcoeff sparse function yields same results
    as numpy's corrcoeff.
    """
    M_d = np.random.random((matrix_size, matrix_size))
    M_s = csr_matrix(M_d)
    C_d = np.corrcoef(M_d)
    C_s = hcs.corrcoef_sparse(M_s)
    assert np.isclose(C_s, C_d, rtol=0.0001).all()


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_compartments_sparse(matrix_size):
    """
    Checks if the eigenvectors obtained by the sparse method match what is
    returned by the dense method.
    """

    M_d = np.random.random((matrix_size, matrix_size))
    M_s = csr_matrix(M_d)
    pc1_d, pc2_d = hcs.compartments(M_d, normalize=False)
    pc1_s, pc2_s = hcs.compartments_sparse(M_s, normalize=False)
    print(pc1_s)
    print(pc1_d)
    assert np.isclose(np.abs(pc1_d), np.abs(pc1_s), rtol=0.01).all()
    assert np.isclose(np.abs(pc2_d), np.abs(pc2_s), rtol=0.01).all()
