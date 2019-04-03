# Tests for the hicstuff digest module
# 20190402
from tempfile import NamedTemporaryFile


def test_write_frag_info():
    """Test generation of fragments_list.txt and info_contigs.txt"""
    ...


def test_attribute_fragments():
    """Test the attribution of reads to restriction fragments"""
    idx_pairs = NamedTemporaryFile(delete=False)
    restriction_table = {}


def test_frag_len():
    """Test the visualisation of fragment lengths distribution"""
    ...
