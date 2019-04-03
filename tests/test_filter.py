# Tests for the hicstuff filter module.
# 20190402

from hicstuff import filter as hcf
from tempfile import NamedTemporaryFile
import hashlib
import os


def hash_file(filename):
    """Computes the MD5 hash of a file's content"""
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def test_get_threshold():
    """Test estimation of filtering threshold and figure generation."""
    fig_file = "test_event_dist.png"
    thr_uncut, thr_loop = hcf.get_thresholds(
        "test_data/valid_idx.pairs", plot_events=True, fig_path=fig_file
    )
    # Check if the estimated threshold are identical to expected values.
    assert thr_uncut == 6
    assert thr_loop == 5

    # Only run if figure works on system 
    if os.path.isfile(fig_file):
        # Compare the hash of the generated figure to an expected hash
        # (generated on the same data)
        fig_hash = hash_file(fig_file)
        assert fig_hash == "7395ee68c940ecf635ceb2b63840568a"
    # Monkey-patch input to simulate stdin when testing interactive mode
    input_values = [6, 5]
    hcf.input = lambda x: input_values.pop(0)
    thr_uncut, thr_loop = hcf.get_thresholds(
        "test_data/valid_idx.pairs",
        interactive=True,
        plot_events=True,
        fig_path=fig_file,
    )
    # Only run if figure generation works on system
    if os.path.isfile(fig_file):
        # Check if hash of plot generated in interactive mode == expected hash
        interact_fig_hash = hash_file(fig_file)
        assert interact_fig_hash == "7395ee68c940ecf635ceb2b63840568a"
        os.remove(fig_file)


def test_filter_pairs():
    """Test pairs file filtering function."""
    filt_pairs = NamedTemporaryFile("w", delete=False)
    filt_pairs.close()
    fig_file = "test_piechart.png"

    hcf.filter_events(
        "test_data/valid_idx.pairs",
        filt_pairs.name,
        6,
        5,
        plot_events=True,
        fig_path=fig_file,
    )

    # Test if the filtered pairs file mathes expectations
    assert hash_file("test_data/valid_idx_filtered.pairs") == hash_file(filt_pairs.name)

    # Only run if figure generation works on system
    if os.path.isfile(fig_file):
        # Test if the hash of generated figures == expected figure
        fig_hash = hash_file(fig_file)
        assert fig_hash == "07cf03c0538728bae8ba5378007b79b3"
        os.unlink(filt_pairs.name)
        os.remove(fig_file)
