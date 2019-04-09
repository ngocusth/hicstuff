# Tests for the hicstuff filter module.
# 20190409

import hicstuff.distance_law as hcdl
import pandas as pd

fragments_file = "test_data/fragments_list.txt"
fragments = pd.read_csv(fragments_file, sep="\t", header=0, usecols=[0, 1, 2, 3])
centro_file = "test_data/centromeres.txt"


def test_get_chr_segment_bins_index():
    """Test getting the index values of the starting positions of the 
    arm/chromosome."""
    # Test with centromeres positions
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments, centro_file)
    assert chr_segment_bins == [0, 129, 409, 474]
    # Test without centromeres positions
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments)
    assert chr_segment_bins == [0, 409]


def test_get_names():
    """Test getting names from a fragment file function."""
    # Test with the centromers option
    names = hcdl.get_names(fragments, [0, 200, 409, 522])
    assert names == ["seq1_left", "seq1_rigth", "seq2_left", "seq2_rigth"]
    # Test without the centromers option
    names = hcdl.get_names(fragments, [0, 409])
    assert names == ["seq1", "seq2"]

