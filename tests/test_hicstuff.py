#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hicstuff testing

Basic tests for functions in the hicstuff library.
"""

import numpy as np
import pytest
import hicstuff as hcs

from inspect import signature
import types

SIZE_PARAMETERS = ("matrix_size", [5, 10, 20, 50, 100])


@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_scn(matrix_size):
    """Test SCN normalization
    
    Check whether a SCN-normalized matrix has all vectors
    summing to one. 
    """

    M = np.random.random((matrix_size, matrix_size))
    M += M.T
    N = hcs.normalize_dense(M, "SCN", iterations=50)
    assert np.isclose(N.sum(axis=1), np.ones(matrix_size), rtol=0.0001).all()


@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_basic_one_argument_functions(M):
    """Check all functions

    Generate an NxN matrix at random and feed it to all functions
    with only one argument. This is meant to catch very fundamental
    errors and also facilitate runtime type guessing with MonkeyType.

    Parameters
    ----------
    M : numpy.ndarray
        A random matrix.
    """
    pass
