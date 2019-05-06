# Tests for the hicstuff filter module.
# 20190409

import hicstuff.distance_law as hcdl
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np
import os as os
import hashlib as hashlib

fragments_file = "test_data/fragments_list.txt"
fragments = pd.read_csv(fragments_file, sep="\t", header=0, usecols=[0, 1, 2, 3])
centro_file = "test_data/centromeres.txt"
pairs_reads_file = "test_data/valid_idx_filtered.pairs"
distance_law_file = "test_data/distance_law.txt"
test_xs, test_ps, labels = hcdl.import_distance_law(distance_law_file)


def hash_file(filename):
    """Computes the MD5 hash of a file's content"""
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def test_import_distance_law():
    """Test importing distance law table files"""
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000])
    assert np.all(np.isclose(test_xs[0], xs[0], rtol=0.0001))
    assert np.all(np.isclose(test_xs[1], xs[1], rtol=0.0001))
    assert len(test_ps) == 2 and len(labels) == 2 and len(test_xs) == len(test_ps)
    assert (
        sum(test_ps[0]) == 3.0341050807866947e-05
        and sum(test_ps[1]) == 0.00010561980134394403
    )
    assert list(np.unique(labels[1])) == ["seq2"]


def test_get_chr_segment_bins_index():
    """Test getting the index values of the starting positions of the 
    arm/chromosome."""
    # Test with centromeres positions.
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments, centro_file)
    assert chr_segment_bins == [0, 129, 409, 474]
    # Test without centromeres positions.
    chr_segment_bins = hcdl.get_chr_segment_bins_index(fragments)
    assert chr_segment_bins == [0, 409]


def test_get_chr_segment_length():
    """Test getting the length of the arms/chromosomes."""
    chr_length = hcdl.get_chr_segment_length(fragments, [0, 129, 409, 474])
    assert chr_length == [19823, 40177, 9914, 10086]


def test_logbins_xs():
    """Test of the function making the logbins."""
    # Test with default values.
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000])
    assert len(xs) == 2
    assert np.all(
        xs[0] == np.unique(np.logspace(0, 115, num=116, base=1.1, dtype=np.int))
    )
    # Test changing base.
    xs = hcdl.logbins_xs(fragments, [0, 409], [60000, 20000], base=1.5)
    assert np.all(
        xs[0] == np.unique(np.logspace(0, 27, num=28, base=1.5, dtype=np.int))
    )
    # Test with the circular option.
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
    # Create a temporary file.
    distance_law = NamedTemporaryFile("w", delete=False)
    # Test with default parameters.
    hcdl.get_distance_law(pairs_reads_file, fragments_file, out_file=distance_law.name)
    assert hash_file(distance_law.name) == hash_file("test_data/distance_law.txt")
    # Test the circular option.
    hcdl.get_distance_law(
        pairs_reads_file, fragments_file, out_file=distance_law.name, circular=True
    )
    assert hash_file(distance_law.name) == "495d9c7ccd7edc33441a6bd5d6fcfc1e"
    # Test the centromere option.
    hcdl.get_distance_law(
        pairs_reads_file,
        fragments_file,
        centro_file=centro_file,
        out_file=distance_law.name,
    )
    assert hash_file(distance_law.name) == "2d31232f2857460d6236d26d33472496"
    # Unlink the temporary file
    os.unlink(distance_law.name)


def test_normalize_distance_law():
    """Test function making the average of distance law."""
    normed_ps = hcdl.normalize_distance_law(test_xs, test_ps)
    assert len(normed_ps) == 2
    assert np.isclose(sum(normed_ps[0]), 5.4540, rtol=0.0001) and np.isclose(
        sum(normed_ps[1]), 27.1305, rtol=0.0001
    )
    assert np.isclose(np.std(normed_ps[0]), 0.1449, rtol=0.001) and np.isclose(
        np.std(normed_ps[1]), 1.9520, rtol=0.001
    )


def test_average_distance_law():
    """Test function making the average of distance law."""
    average_xs, average_ps = hcdl.average_distance_law(test_xs, test_ps, None)
    assert np.all(average_xs == test_xs[0])
    assert np.isclose(sum(average_ps), 6.8119e-05, rtol=10e-7)
    assert np.isclose(np.std(average_ps), 3.5803e-06, rtol=10e-7)


def test_slope_distance_law():
    """Test function calculating the slope of the distance law."""
    slope = hcdl.slope_distance_law(test_xs, test_ps)
    assert len(slope) == 2
    assert np.isclose(sum(slope[0]), 18.9329, rtol=0.0001) and np.isclose(
        sum(slope[1]), -2.7459, rtol=0.0001
    )
    assert np.isclose(np.std(slope[0]), 3.9226, rtol=0.0001) and np.isclose(
        np.std(slope[1]), 5.0451, rtol=0.0001
    )
