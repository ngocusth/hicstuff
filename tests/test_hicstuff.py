#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hicstuff testing

Basic tests for functions in the hicstuff library.
"""

import random
import numpy as np
import pytest
import warnings
import hicstuff.hicstuff as hcs
from scipy.sparse import coo_matrix
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
    M : numpy.ndarray
        A random matrix.
    """
    functions_list = getmembers(hcs, isfunction)
    M_d = np.random.random((matrix_size, matrix_size))
    M_d += M_d.T
    for _, func in functions_list:
        params = signature(func).parameters
        annot = params[list(params.keys())[0]].annotation
        # TODO: add functions annotation to use this test
        if len(params) == 1 and annot == np.array:
            assert func(M_d).any()
