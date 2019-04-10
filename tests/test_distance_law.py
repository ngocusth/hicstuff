# Tests for the hicstuff filter module.
# 20190409

import hicstuff.distance_law as hcdl
from tempfile import NamedTemporaryFile
from test_filter import hash_file
import pandas as pd
import numpy as np
import os as os

# fragments_file = "test_data/fragments_list.txt"
# fragments = pd.read_csv(fragments_file, sep="\t", header=0, usecols=[0, 1, 2, 3])
# centro_file = "test_data/centromeres.txt"
# pairs_reads_file = "test_data/valid_idx_filtered.pairs"


def test_get_chr_segment_bins_index():
    """Test getting the index values of the starting positions of the 
    arm/chromosome."""
    # Test with centromeres positions
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments, centro_file)
    assert chr_segment_bins == [0, 129, 409, 474]
    # Test without centromeres positions
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments)
    assert chr_segment_bins == [0, 409]


def test_get_chr_segment_length():
    """Test getting the length of the arms/chromosomes."""
    chr_length = hcdl.get_chr_segment_length(fragments, [0, 129, 409, 474])
    assert chr_length == [19823, 40177, 9914, 10086]


def test_logbins_xs():
    """Test of the function making the logbins."""
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000])
    assert len(xs) == 2
    assert np.all(
        xs[0] == np.unique(np.logspace(0, 115, num=116, base=1.1, dtype=np.int))
    )
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000], base=1.5)
    assert np.all(
        xs[0] == np.unique(np.logspace(0, 27, num=28, base=1.5, dtype=np.int))
    )
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000], circular=True)
    assert np.all(
        xs[0] == np.unique(np.logspace(0, 108, num=109, base=1.1, dtype=np.int))
    )


def test_get_names():
    """Test getting names from a fragment file function."""
    # Test with the centromers option
    names = hcdl.get_names(fragments, [0, 200, 409, 522])
    assert names == ["seq1_left", "seq1_rigth", "seq2_left", "seq2_rigth"]
    # Test without the centromers option
    names = hcdl.get_names(fragments, [0, 409])
    assert names == ["seq1", "seq2"]


def test_get_distance_law():
    """Test the general distance_law function."""
    distance_law = NamedTemporaryFile("w", delete=False)
    hcdl.get_distance_law(pairs_reads_file, fragments_file, outdir=distance_law.name)
    assert hash_file(distance_law.name) == hash_file("/test_data/distance_law.txt")
    hcdl.get_distance_law(
        pairs_reads_file, fragments_file, outdir=distance_law.name, circular=True
    )
    assert hash_file(distance_law.name) == "495d9c7ccd7edc33441a6bd5d6fcfc1e"
    hcdl.get_distance_law(
        pairs_reads_file,
        fragments_file,
        centro_file=centro_file,
        outdir=distance_law.name,
    )
    assert hash_file(distance_law.name) == "2d31232f2857460d6236d26d33472496"
    os.unlink(distance_law.name)

