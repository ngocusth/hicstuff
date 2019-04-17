# Test functions for the pipeline submodule

from tempfile import NamedTemporaryFile
import os
import pandas as pd
import filecmp
import numpy as np
import hicstuff.pipeline as hpi


def test_sam2pairs():
    ...


def test_pairs2mat():
    ...


def test_filter_pcr_dup():
    """Test if PCR duplicates are removed correctly"""
    dup_pairs = NamedTemporaryFile(mode="w", delete=False)
    dup_rm = NamedTemporaryFile(mode="w", delete=False)
    lnum = 0
    # Copy the test valid_idx file, but generate PCR dups of the pair at line 50
    with open("test_data/valid_idx.pairs", "r") as pairs:
        for line in pairs:
            dup_pairs.write(line)
            if lnum == 50:
                # Making 30 duplicates of this pair
                for i in range(30):
                    dup_pairs.write(line)
            lnum += 1
    dup_pairs.close()
    dup_rm.close()

    # Remove duplicates
    hpi.filter_pcr_dup(dup_pairs.name, dup_rm.name)

    # Check if duplicates have been removed correctly
    assert filecmp.cmp("test_data/valid_idx.pairs", dup_rm.name)
    os.unlink(dup_pairs.name)
    os.unlink(dup_rm.name)


def test_full_pipeline():
    ...
