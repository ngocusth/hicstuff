# hicstuff

[![PyPI version](https://badge.fury.io/py/hicstuff.svg)](https://badge.fury.io/py/hicstuff)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hicstuff.svg)
[![Build Status](https://travis-ci.com/koszullab/hicstuff.svg)](https://travis-ci.com/koszullab/hicstuff)
[![codecov](https://codecov.io/gh/koszullab/hicstuff/branch/master/graph/badge.svg)](https://codecov.io/gh/koszullab/hicstuff)
[![Read the docs](https://readthedocs.org/projects/hicstuff/badge)](https://hicstuff.readthedocs.io)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

A lightweight library that generates and handles Hi-C contact maps in either cooler-compatible 2Dbedgraph or [instaGRAAL](https://github.com/koszullab/instaGRAAL) format. It is essentially a merge of the [yahcp](https://github.com/baudrly/yahcp) pipeline, the [hicstuff](https://github.com/baudrly/hicstuff) library and extra features illustrated in the [3C tutorial](https://github.com/axelcournac/3C_tutorial) and the [DADE pipeline](https://github.com/scovit/dade), all packaged together for extra convenience.

The goal is to make generation and manipulation of Hi-C matrices as simple as possible and work for any organism.

## Table of contents

* [Installation](#Installation)
* [Usage](#Usage)
  * [Full pipeline](#Full-pipeline)
  * [Individual components](#Individual-components)
* [Library](#Library)
* [Connecting the modules](#Connecting-the-modules)
* [File formats](#File-formats)

## Installation

To install a stable version:
```sh
   pip3 install -U hicstuff
```

or, for the latest development version:

```sh
    pip3 install -e git+https://github.com/koszullab/hicstuff.git@master#egg=hicstuff
```

## Usage

### Full pipeline

All components of the pipelines can be run at once using the `hicstuff pipeline` command. This allows to generate a contact matrix from reads in a single command. By default, the output sparse matrix is in GRAAL format, but it can be a 2D bedgraph file if required.

    usage:
        pipeline [--quality-min=INT] [--size=INT] [--no-cleanup] [--start-stage=STAGE]
                 [--threads=INT] [--minimap2] [--matfmt=FMT] [--prefix=PREFIX]
                 [--tmpdir=DIR] [--iterative] [--outdir=DIR] [--filter]
                 [--enzyme=ENZ] [--plot] [--circular] --fasta=FILE <input1> [<input2>]

    arguments:
        input1:             Forward fastq file, if start_stage is "fastq", sam
                            file for aligned forward reads if start_stage is
                            "sam", or a .pairs file if start_stage is "pairs".
        input2:             Reverse fastq file, if start_stage is "fastq", sam
                            file for aligned reverse reads if start_stage is
                            "sam", or nothing if start_stage is "pairs".


    options:
        -M, --matfmt=FMT              The format of the output sparse matrix.
                                      Can be "cooler" for 2D Bedgraph format 
                                      compatible with cooler, or "GRAAL" for
                                      GRAAL-compatible format. [default: GRAAL]
        -C, --circular                Enable if the genome is circular.
        -e, --enzyme=ENZ              Restriction enzyme if a string, or chunk
                                      size (i.e. resolution) if a number. Can
                                      also be multiple comma-separated enzymes.
                                      [default: 5000]
        -f, --fasta=FILE              Reference genome to map against in FASTA
                                      format
        -F, --filter                  Filter out spurious 3C events (loops and
                                      uncuts) using hicstuff filter. Requires
                                      "-e" to be a restriction enzyme, not a
                                      chunk size.
        -S, --start-stage=STAGE       Define the starting point of the pipeline
                                      to skip some steps. Default is "fastq" to
                                      run from the start. Can also be "sam" to
                                      skip the alignment, pairs to start from a
                                      singl pairs file or pairs_idx to skip
                                      fragment attribution and only build the 
                                      matrix. [default: fastq]
        -i, --iterative               Map reads iteratively using hicstuff
                                      iteralign, by truncating reads to 20bp
                                      and then repeatedly extending and
                                      aligning them.
        -m, --minimap2                Use the minimap2 aligner instead of
                                      bowtie2. Not enabled by default.
        -n, --no-cleanup              If enabled, intermediary BED files will
                                      be kept after generating the contact map.
                                      Disabled by defaut.
        -o, --outdir=DIR              Output directory. Defaults to the current
                                      directory.
        -p, --plot                    Generates plots in the output directory
                                      at different steps of the pipeline.
        -P, --prefix=PREFIX           Overrides default GRAAL-compatible
                                      filenames and use a prefix with
                                      extensions instead.
        -q, --quality-min=INT         Minimum mapping quality for selecting
                                      contacts. [default: 30].
        -s, --size=INT                Minimum size threshold to consider
                                      contigs. Keep all contigs by default.
                                      [default: 0]
        -t, --threads=INT             Number of threads to allocate.
                                      [default: 1].
        -T, --tmpdir=DIR              Directory for storing intermediary BED
                                      files and temporary sort files. Defaults
                                      to the output directory.

    output:
        abs_fragments_contacts_weighted.txt: the sparse contact map
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes
        hicstuff.log: details and statistics about the run.


For example, to run the pipeline with minimap2 using 8 threads and generate a matrix in instagraal format in the directory `out`:
```
hicstuff pipeline -t 8 -m -e DpnII -o out/ -f genome.fa reads_for.fq reads_rev.fq
```

The pipeline can also be run from python, using the `hicstuff.pipeline` submodule. For example, this would run the pipeline with bowtie2 (default) using iterative alignment and keep all intermediate files.

```python
from hicstuff import pipeline as hpi

hpi.full_pipeline(
    'genome.fa', 
    'end1.fq', 
    'end2.fq', 
    no_cleanup=True
    iterative=True
    out_dir='out', 
    enzyme="DpnII")
```

The general steps of the pipeline are as follows:

![Pipeline flowchart](doc/images/pipeline.svg)

### Individual components

For more advanced usage, different scripts can be used independently on the command line to perform individual parts of the pipeline. This readme contains quick descriptions and example usages. To obtain detailed instructions on any subcommand, one can use `hicstuff <subcommand> --help`.

#### Iterative alignment

Truncate reads from a fastq file to 20 basepairs and iteratively extend and re-align the unmapped reads to optimize the proportion of uniquely aligned reads in a 3C library.

    usage:
        hicstuff iteralign [--minimap2] [--threads=1] [--min_len=20] --out_sam=FILE --fasta=FILE <reads.fq>

#### Digestion of the genome

Digests a fasta file into fragments based on a restriction enzyme or a
fixed chunk size. Generates two output files into the target directory
named "info_contigs.txt" and "fragments_list.txt"

    usage:
        digest [--plot] [--figdir=FILE] [--circular] [--size=INT]
               [--outdir=DIR] --enzyme=ENZ <fasta>

 
 For example, to digest the yeast genome with MaeII and HinfI and show histogram of fragment lengths:

`hicstuff digest --plot --outdir output_dir --enzyme MaeII,HinfI Sc_ref.fa`

#### Filtering of 3C events

Filters spurious 3C events such as loops and uncuts from the library based
on a minimum distance threshold automatically estimated from the library by default. Can also plot 3C library statistics.

    usage:
        filter [--interactive | --thresholds INT-INT] [--plot]
               [--figdir FILE] <input> <output>

#### Viewing the contact map

Visualize a Hi-C matrix file as a heatmap of contact frequencies. Allows to tune visualisation by binning and normalizing the matrix, and to save the output image to disk. If no output is specified, the output is displayed interactively. If two contact maps are provided, the log ratio of the first divided by the second will be shown.

    usage:
        view [--binning=1] [--despeckle] [--frags FILE] [--trim INT]
             [--normalize] [--max=99] [--output=IMG] [--cmap=CMAP]
             [--log] [--region=STR] <contact_map> [<contact_map2>]

For example, to view a 1Mb region of chromosome 1 from a full genome Hi-C matrix rebinned at 10kb:

```sh
    hicstuff view --normalize --binning 10kb --region chr1:10,000,000-11,000,000 --frags fragments_list.txt contact_map.tsv
```
### Library

All components of the hicstuff program can be used as python modules. See the documentation on [reathedocs](https://hicstuff.readthedocs.io). The expected contact map format for the library is a simple CSV file, and the objects handled by the library are simple ```numpy``` arrays. The various submodules of hicstuff contain various utilities.

```python
import hicstuff.digest # Functions to work with restriction fragments
import hicstuff.iteralign # Functions related to iterative alignment
import hicstuff.hicstuff # Contains utilities to modify and operate on contact maps as numpy arrays
import hicstuff.filter # Functions for filtering 3C events by type (uncut, loop)
import hicstuff.view # Utilities to visualise contact maps
import hicstuff.io # Reading and writing hicstuff files
import hicstuff.pipeline # Generation and processing of files to generate matrices.
```

### Connecting the modules

All the steps described here are handled automatically when running the `hicstuff pipeline`. But if you want to connect the different modules manually, the intermediate input and output files must be processed using light python scripting.

#### Aligning the reads

You can generate SAM files independently using your favorite read mapping software, use the command line utility `hicstuff iteralign`, or use the helper function `align_reads` in the submodule `hicstuff.pipeline`. For example, to perform iterative alignment using minimap2 (instead of bowtie2 by default):

**Using the python function:**

```python
from hicstuff import pipeline as hpi

hpi.align_reads("end1.fastq", "genome.fasta", "end1.sam", iterative=True, minimap2=True)
```

**Using the command line tool:**

```bash
hicstuff iteralign --minimap2 --iterative -f genome.fasta -o end1.sam end1.fastq
```


#### Extracting contacts from the alignment

The output from `hicstuff iteralign` is a SAM file. In order to retrieve Hi-C pairs, you need to run iteralign separately on the two fastq files and process the resulting alignment files as follows using the `pipeline` submodules of hicstuff.

```python
from hicstuff import pipeline as hpi
import pysam as ps
# Sort alignments by read names
ps.sort("-n", "-O", "SAM", "-o", "end1.sam.sorted", "end1.sam")
ps.sort("-n", "-O", "SAM", "-o", "end2.sam.sorted", "end2.sam")
# Combine SAM files
hpi.sam2pairs("end1.sorted.sam", "end2.sorted.sam", "output.pairs", "info_contigs.txt", min_qual=30)

```
This will generate a "pairs" file containing all read pairs where both reads have been aligned with a mapping quality of at least 30.

#### Attributing each read to a restriction fragment
To build a a contact matrix, we need to attribute each read to a fragment in the genome. This is done under the hood by performing a binary search for each read position against the list of restriction sites in the genome.

```python
from hicstuff import digest as hcd
from Bio import SeqIO

# Build a list of restriction sites for each chromosome
restrict_table = {}
for record in SeqIO.parse("genome.fasta", "fasta"):
    # Get chromosome restriction table
    restrict_table[record.id] = hcd.get_restriction_table(
        record.seq, enzyme, circular=circular
    )

# Add fragment index to pairs (readID, chr1, pos1, chr2,
# pos2, strand1, strand2, frag1, frag2)
hcd.attribute_fragments("output.pairs", "output_indexed.pairs", restrict_table)

```

#### Filtering pairs
The resulting pairs file can then be filtered, either in the command line using the `hicstuff filter` command, or in python using the `hicstuff.filter` submodule. Otherwise, the matrix can be built directly from the unfiltered pairs. 

**Filtering on the command line:**
```bash
hicstuff filter output_indexed.pairs output_filtered.pairs
```
**Filtering in python:**
```python
from hicstuff import filter as hcf

uncut_thr, loop_thr = hcf.get_thresholds("output_indexed.pairs")
hcf.filter_events("output_indexed.pairs", "output_filtered.pairs", uncut_thr, loop_thr)
```
Note that both the command and the python function have various options to generate figure or tweak the filtering thresholds. These options can be displayed using `hicstuff filter -h`

#### Matrix generation
A Hi-C sparse contact matrix can then be generated using the python submodule `hicstuff.pipeline`. The matrix can be generated either in GRAAL format, or bedgraph2 (WIP) for cooler compatibility.

```python
from hicstuff import pipeline as hpi

n_frags = sum(1 for line in open(fragments_list, "r")) - 1
hpi.pairs2matrix("output_filtered.pairs", "abs_fragments_contacts_weighted.txt", 'fragments_list.txt', mat_fmt="GRAAL")
```

### File formats

* pairs files: This format is used for all intermediate files in the pipeline and is also used by `hicstuff filter`. It is a space-separated format holding informations about Hi-C pairs. It has an [official specification](https://github.com/4dn-dcic/pairix/blob/master/pairs_format_specification.md) defined by the 4D Nucleome data coordination and integration center.
* 2D bedgraph: This is an optional output format of `hicstuff pipeline` for the sparse matrix. It has two fragment per line, and the number of times they are found together. It has the following fields: **chr1, start1, end1, chr2, start2, end2, occurences**
    - Those files can be [loaded by cooler](https://cooler.readthedocs.io/en/latest/cli.html?highlight=load#cooler-load) using `cooler load -f bg2 <chrom.sizes>:<binsize> in.bg2.gz out.cool` where chrom.sizes is a tab delimited file with chromosome names and length on each line, and binsize is the size of bins in the matrix.
* GRAAL sparse matrix: This is a simple tab-separated file with 3 columns: **frag1, frag2, contacts**. The id columns correspond to the absolute id of the restriction fragments (0-indexed). The first row is a header containing the number of rows, number of columns and number of nonzero entries in the matrix. Example:

```
564	564	6978
0	0	3
1	2	4
1	3	3

```

* fragments_list.txt: This tab separated file provides information about restriction fragments positions, size and GC content. Note the coordinates are 0 point basepairs, unlike the pairs format, which has 1 point basepairs. Example:
   - id: 1 based restriction fragment index within chromosome.
   - chrom: Chromosome identifier. Order should be the same as in info_contigs.txt or pairs files.
   - start_pos: 0-based start of fragment, in base pairs.
   - end_pos: 0-based end of fragment, in base pairs.
   - size: Size of fragment, in base pairs.
   - gc_content: Proportion of G and C nucleotide in the fragment.
```
id	chrom	start_pos	end_pos	size	gc_content
1	seq1	0	21	21	0.5238095238095238
2	seq1	21	80	59	0.576271186440678
3	seq1	80	328	248	0.5201612903225806
```

* info_contigs.txt: This tab separated file gives information on contigs, such as number of restriction fragments and size. Example:
   - contig: Chromosome identified. Order should be the same in pairs files or fragments_list.txt.
   - length: Chromosome length, in base pairs.
   - n_frags: Number of restriction fragments in chromosome.
   - cumul_length: Cumulative length of previous chromosome, in base pairs.

```
contig	length	n_frags	cumul_length
seq1	60000	409	0
seq2	20000	155	409
```

