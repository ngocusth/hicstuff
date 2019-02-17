# hicstuff

[![PyPI version](https://badge.fury.io/py/hicstuff.svg)](https://badge.fury.io/py/hicstuff)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hicstuff.svg)
[![Build Status](https://travis-ci.com/koszullab/hicstuff.svg)](https://travis-ci.com/koszullab/hicstuff)
[![Read the docs](https://readthedocs.org/projects/hicstuff/badge)](https://hicstuff.readthedocs.io)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

A lightweight library that generates and handles Hi-C contact maps in either 2Dbedgraph or [instaGRAAL](https://github.com/koszullab/instaGRAAL) format. It is essentially a merge of the [yahcp](https://github.com/baudrly/yahcp) pipeline, the [hicstuff](https://github.com/baudrly/hicstuff) library and extra features illustrated in the [3C tutorial](https://github.com/axelcournac/3C_tutorial) and the [DADE pipeline](https://github.com/scovit/dade), all packaged together for extra convenience.

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

Note for OSX and BSD users: `hicstuff pipeline` relies on the GNU coreutils. If you want to use it, you should use these as your default. [Here](https://www.topbug.net/blog/2013/04/14/install-and-use-gnu-command-line-tools-in-mac-os-x/) is a tutorial to set the gnu coreutils as default commands on OSX. 

## Usage

### Full pipeline

All components of the pipelines can be run at once using the `hicstuff pipeline` commands. This allows to generate a contact matrix from reads in a single command. By default, the output sparse matrix is in GRAAL format, but it can be a 2D bedgraph file if required.

    usage:
        pipeline [--quality_min=INT] [--duplicates] [--size=INT] [--no-cleanup]
                 [--threads=INT] [--minimap2] [--bedgraph] [--prefix=PREFIX]
                 [--tmpdir=DIR] [--iterative] [--outdir=DIR] [--filter]
                 [--enzyme=ENZ] [--plot] --fasta=FILE
                 (<fq1> <fq2> | --sam <sam1> <sam2> | --pairs <bed2D>)

    arguments:
        fq1:             Forward fastq file. Required by default.
        fq2:             Reverse fastq file. Required by default.
        sam1:            Forward SAM file. Required if using --sam to skip
                         mapping.
        sam2:            Reverse SAM file. Required if using --sam to skip
                         mapping.
        bed2D:           Sorted 2D BED file of pairs. Required if using
                         "--pairs" to only build matrix.


    options:
        -b, --bedgraph                If enabled, generates a sparse matrix in
                                      2D Bedgraph format (cooler-compatible)
                                      instead of GRAAL-compatible format.
        -C, --circular                Enable if the genome is circular.
        -d, --duplicates:             If enabled, trims (10bp) adapters and
                                      remove PCR duplicates prior to mapping.
                                      Only works if reads start with a 10bp
                                      sequence. Not enabled by default.
        -e ENZ, --enzyme=ENZ          Restriction enzyme if a string, or chunk
                                      size (i.e. resolution) if a number. Can
                                      also be multiple comma-separated enzymes.
                                      [default: 5000]
        -f FILE, --fasta=FILE         Reference genome to map against in FASTA
                                      format
        -F, --filter                  Filter out spurious 3C events (loops and
                                      uncuts) using hicstuff filter. Requires
                                      "-e" to be a restriction enzyme, not a
                                      chunk size.
        -S, --sam                     Skip the mapping and start pipeline from
                                      fragment attribution using SAM files.
        -i, --iterative               Map reads iteratively using hicstuff
                                      iteralign, by truncating reads to 20bp
                                      and then repeatedly extending and
                                      aligning them.
        -m, --minimap2                Use the minimap2 aligner instead of
                                      bowtie2. Not enabled by default.
        -A, --pairs                   Start from the matrix building step using
                                      a sorted list of pairs in 2D BED format.
        -n, --no-cleanup              If enabled, intermediary BED files will
                                      be kept after generating the contact map.
                                      Disabled by defaut.
        -o DIR, --outdir=DIR          Output directory. Defaults to the current
                                      directory.
        -p, --plot                    Generates plots in the output directory
                                      at different steps of the pipeline.
        -P PREFIX, --prefix=PREFIX    Overrides default GRAAL-compatible
                                      filenames and use a prefix with
                                      extensions instead.
        -q INT, --quality_min=INT     Minimum mapping quality for selecting
                                      contacts. [default: 30].
        -s INT, --size=INT            Minimum size threshold to consider
                                      contigs. Keep all contigs by default.
                                      [default: 0]
        -t INT, --threads=INT         Number of threads to allocate.
                                      [default: 1].
        -T DIR, --tmpdir=DIR          Directory for storing intermediary BED
                                      files and temporary sort files. Defaults
                                      to the output directory.

    output:
        abs_fragments_contacts_weighted.txt: the sparse contact map
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes

For example, to run the pipeline with minimap2 using 8 threads and generate a matrix in instagraal format in the directory `out`:
```
hicstuff pipeline -t 8 -m -e DpnII -o out/ -f genome.fa reads_for.fq reads_rev.fq
```

### Individual components

For more advanced usage, different scripts can be used independently on the command line to perform individual parts of the pipeline.

#### Iterative alignment

Truncate reads from a fastq file to 20 basepairs and iteratively extend and re-align the unmapped reads to optimize the proportion of uniquely aligned reads in a 3C library.

    usage:
        hicstuff iteralign [--minimap2] [--threads=1] [--min_len=20] --out_sam=FILE --fasta=FILE <reads.fq>

    arguments:
        reads.fq                Fastq file containing the reads to be aligned

    options:
        -f FILE, --fasta=FILE   Fasta file on which to map the reads.
        -t INT, --threads=INT  Number of parallel threads allocated for the alignment [default: 1].
        -T DIR, --tempdir=DIR  Temporary directory. Defaults to current directory.
        -m, --minimap2     If set, use minimap2 instead of bowtie2 for the alignment.
        -l INT, --min_len=INT  Length to which the reads should be truncated [default: 20].
        -o FILE, --out_sam=FILE Path where the alignment will be written in SAM format.

#### Digestion of the genome

Digests a fasta file into fragments based on a restriction enzyme or a
fixed chunk size. Generates two output files into the target directory
named "info_contigs.txt" and "fragments_list.txt"

    Digests a fasta file into fragments based on a restriction enzyme or a
    fixed chunk size. Generates two output files into the target directory
    named "info_contigs.txt" and "fragments_list.txt"

    usage:
        digest [--plot] [--figdir=FILE] [--circular] [--size=INT]
               [--outdir=DIR] --enzyme=ENZ <fasta>

    arguments:
        fasta                     Fasta file to be digested

    options:
        -c, --circular                  Specify if the genome is circular.
        -e, --enzyme=ENZ[,ENZ2,...]     A restriction enzyme or an integer
                                        representing fixed chunk sizes (in bp).
                                        Multiple comma-separated enzymes can
                                        be given.
        -s INT, --size=INT              Minimum size threshold to keep
                                        fragments. [default: 0]
        -o DIR, --outdir=DIR            Directory where the fragments and
                                        contigs files will be written.
                                        Defaults to current directory.
        -p, --plot                      Show a histogram of fragment length
                                        distribution after digestion.
        -f FILE, --figdir=FILE          Path to directory of the output figure.
                                        By default, the figure is only shown
                                        but not saved.

    output:
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes

 For example, to digest the yeast genome with MaeII and HinfI and show histogram of fragment lengths:

`hicstuff digest -p -o output_dir -e MaeII,HinfI Sc_ref.fa`

#### Filtering of 3C events

Filters spurious 3C events such as loops and uncuts from the library based
on a minimum distance threshold automatically estimated from the library by default. Can also plot 3C library statistics.

    usage:
        filter [--interactive | --thresholds INT-INT] [--plot]
               [--figdir FILE] <input> <output>

    arguments:
        input       2D BED file containing coordinates of Hi-C interacting
                    pairs, the index of their restriction fragment and their
                    strands.
        output      Path to the filtered file, in the same format as the input.

    options:
        -i, --interactive                 Interactively shows plots and asks
                                          for thresholds.
        -t INT-INT, --thresholds=INT-INT  Manually defines integer values for
                                          the thresholds in the order
                                          [uncut, loop].
        -p, --plot                        Shows plots of library composition
                                          and 3C events abundance.
        -f DIR, --figdir=DIR              Path to the output figure directory.
                                          By default, the figure is only shown
                                          but not saved.

#### Viewing the contact map

Visualize a Hi-C matrix file as a heatmap of contact frequencies. Allows to tune visualisation by binning and normalizing the matrix, and to save the output image to disk. If no output is specified, the output is displayed.

    usage:
        view [--binning=1] [--despeckle] [--frags FILE] [--trim INT]
             [--normalize] [--max=99] [--output=IMG] [--cmap=CMAP]
             [--log] [--region=STR] <contact_map> [<contact_map2>]

    arguments:
        contact_map             Sparse contact matrix in GRAAL format
        contact_map2            Sparse contact matrix in GRAAL format,
                                if given, the log ratio of
                                contact_map/contact_map2 will be shown


    options:
        -b, --binning=INT[bp|kb|Mb|Gb]   Subsampling factor or fix value in
                                         basepairs to use for binning
                                         [default: 1].
        -c, --cmap=CMAP                  The name of a matplotlib colormap to
                                         use for the matrix. [default: Reds]
        -C, --circular                   Use if the genome is circular.
        -d, --despeckle                  Remove sharp increases in long range
                                         contact by averaging surrounding
                                         values.
        -f FILE, --frags=FILE            Required for bp binning. Tab-separated
                                         file with headers, containing
                                         fragments start position in the 3rd
                                         column, as generated by hicstuff
                                         pipeline.
        -l, --log                        Log transform pixel values to improve
                                         visibility of long range signals.
        -m INT, --max=INT                Saturation threshold. Maximum pixel
                                         value is set to this percentile
                                         [default: 99].
        -n, --normalize                  Should SCN normalization be performed
                                         before rendering the matrix ?
        -o IMG, --output=IMG             Path where the matrix will be stored
                                         in PNG format.
        -r STR[;STR], --region=STR[;STR] Only view a region of the contact map.
                                         Regions are specified as UCSC strings.
                                         (e.g.:chr1:1000-12000). If only one
                                         region is given, it is viewed on the
                                         diagonal. If two regions are given,
                                         The contacts between both are shown.
        -t INT, --trim=INT               Trims outlier rows/columns from the
                                         matrix if the sum of their contacts
                                         deviates from the mean by more than
                                         INT standard deviations.

For example, to view a 1Mb region of chromosome 1 from a full genome Hi-C matrix rebinned at 10kb:

```sh
    hicstuff view -n -b 10kb -r chr1:10,000,000-11,000,000 -f fragments_list.txt contact_map.tsv
```
### Library

All components of the hicstuff program can be used as python modules. See the documentation on [reathedocs](https://hicstuff.readthedocs.io). The expected contact map format for the library is a simple CSV file, and the objects handled by the library are simple ```numpy``` arrays. The various submodules of hicstuff contain various utilities.

```python
import hicstuff.digest # Functions to work with fragments (digestion, matrix building)
import hicstuff.iteralign # Functions related to iterative alignment
import hicstuff.hicstuff # Contains utilities to modify and operate on contact maps as numpy arrays
import hicstuff.filter # Functions for filtering 3C events by type (uncut, loop)
import hicstuff.view # Utilities to visualise contact maps
```

### Connecting the modules

All the steps described here are handled automatically when running the `hicstuff pipeline`. But if you want to connect the different modules manually, the intermediate input and output files must be processed using light bash scripting.

#### Extracting contacts from the alignment

The output from iteralign is a SAM file. In order to retrieve Hi-C pairs, you need to run iteralign separately on the two fastq files and process the resulting alignment files processed as follows using bedtools and some bash commands.

1. Convert the SAM files into BED format

```bash
samtools view -bS -F 260 -@ $t -q 30 "for.sam" |
  bedtools bamtobed -i - |
  awk 'OFS="\t" { print $1,$2,$3,$4,$6 }' \
    > contacts_for.bed

samtools view -bS -F 260 -@ $t -q 30 "rev.sam" |
  bedtools bamtobed -i - |
  awk 'OFS="\t" { print $1,$2,$3,$4,$6 }' \
    > contacts_rev.bed
```

2. Put all forward and reverse reads into a single sorted BED file

```bash
sort -k1,1d -k2,2n contacts_for.bed contacts_rev.bed \
  > total_contacts.bed

```

#### Attributing each read to a restriction fragment
To build a a contact matrix, we need to attribute each read to a fragment in the genome. This is done by intersecting all the reads with the digested genome.

1. Extract restriction fragments from the genome using `hicstuff digest`.

```bash
# Generate fragments_list.txt
hicstuff digest --fasta genome.fa \
  --enzyme DpnII \
  --output-dir .

# Make a BED from it
awk 'NR>1 { print $2"\t"$3"\t"$4"\t"(NR-2) }' fragments_list.txt
  >fragments_list.bed

```

2. Intersect the BED files of reads and restriction fragments.

```bash
bedtools intersect -a total_contacts.bed \
  -b fragments_list.bed -wa -wb |
  sort -k4d  \
  > contact_intersect_sorted.bed
```

4. Get reads into a paired BED file (1 pair per line)

```bash
# Note: F is fragment and R is read
# This awk snippet allows to convert a sorted BED file with fields:
#   Rchr Rstart Rend Rname Rstrand Fchr Fstart Fend Fidx
# to a "2D BED" file with:
#   F1chr F1start F1end F1idx R1strand F2chr F2start F2end F2idx R2strand
bed2pairs='
BEGIN{dir="for"; OFS="\t"}
{
  if(dir=="for") {
    fw["name"]=$4; fw["coord"]=$6"\t"$7"\t"$8"\t"$9"\t"$5; dir="rev" }
  else {
    if(fw["name"] == $4) {
        print fw["coord"],$6,$7,$8,$9,$5; dir="for"}
    else {
        dir="rev"; fw["coord"]=$6"\t"$7"\t"$8"\t"$9"\t"$5; fw["name"]=$4}
    }
}
'
awk "$bed2pairs" contact_intersect_sorted.bed > contact_intersect_sorted.bed2D
```

The resulting 2D BED file can then be filtered by the `hicstuff filter` module if needed, otherwise, the matrix can be built directly from it. To generate a GRAAL-compatible sparse matrix from the 2D bed file:

```bash
# Remove strand information, sort by fragment combination,
# Count occurrences of each fragment combination and format into csv.
echo -e "id_fragment_a\tid_fragment_b\tn_contact" > matrix.tsv
cut -f4,9 "$tmp_dir/contact_intersect_sorted.bed2D" |
  sort -V |
  uniq -c |
  sed 's/^ *//' |
  tr ' ' '\t' |
  awk '{print $0,$1}' |
  cut -f1 --complement >> matrix.tsv
```

### File formats

* 2D BED: This is the input format for `hicstuff filter`. It has one line per Hi-C pair and the following fields for each read: **chr start end name frag_id strand**.
* 2D bedgraph: This is an optional output format of `hicstuff pipeline` for the sparse matrix. It has two fragment per line, and the number of times they are found together. It has the following fields: **chr1, start1, end1, chr2, start2, end2, occurences**

* GRAAL sparse matrix: This is a simple CSV file with 3 columns: **id_a, id_b, occurrences**. The id columns correspond to the absolute id of the restriction fragments (0-indexed).
