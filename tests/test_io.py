# test input/output functions from hicstuff
# 20190402
from tempfile import NamedTemporaryFile
import os
import gzip
import zipfile
import bz2
from hicstuff.io import read_compressed, is_compressed


def test_compress():
    """Test reading and checking of compressed files"""

    # Generate temp files and store path
    f = NamedTemporaryFile(delete=False, mode="w")
    fgz = NamedTemporaryFile(delete=False, mode="wb")
    fbz = NamedTemporaryFile(delete=False)
    fz = NamedTemporaryFile(delete=False)
    fgz.close()
    fbz.close()
    fz.close()
    # Fill with some text
    f.write("xyz")
    f.close()
    # Write text to compressed file using different compression types
    raw = open(f.name, mode="rb").read()
    gz = gzip.open(fgz.name, mode="wb")
    gz.write(raw)
    gz.close()
    bz = bz2.BZ2File(fbz.name, "wb")
    bz.write(raw)
    bz.close()
    z = zipfile.ZipFile(fz.name, "w")
    z.write(f.name)
    z.close()
    exp_compress = {f.name: False, fgz.name: True, fz.name: True, fbz.name: True}
    for fh in [f, fgz, fbz, fz]:
        content = read_compressed(fh.name).read()
        # Check reading
        assert content == "xyz"
        # Check guessing compression state
        assert is_compressed(fh.name) == exp_compress[fh.name]
        # Clean files
        os.unlink(fh.name)
