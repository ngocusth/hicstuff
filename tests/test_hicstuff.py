#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hicstuff testing

Basic tests for functions in the hicstuff library.
"""

import numpy as np
import pytest
import hicstuff as hcs

SIZE_PARAMETERS = ("matrix_size", [5, 10, 20, 50, 100])


@pytest.mark.parametrize(*SIZE_PARAMETERS)
def test_scn(matrix_size):
    M = np.random.random((matrix_size, matrix_size))
    M += M.T
    N = hcs.normalize_dense(M, "SCN", iterations=50)
    assert np.isclose(N.sum(axis=1), np.ones(matrix_size), rtol=0.0001).all()
