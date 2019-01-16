# Structure based on Rémy Greinhofer (rgreinho) tutorial on subcommands in
# docopt : https://github.com/rgreinho/docopt-subcommands-example
# cmdoret, 20181412
from hicstuff.hicstuff import bin_sparse, normalize_sparse, bin_bp_sparse
import re
from hicstuff.iteralign import *
from hicstuff.digest import write_frag_info, write_sparse_matrix, frag_len
from hicstuff.filter import get_thresholds, filter_events, process_read_pair
from hicstuff.view import (
    load_raw_matrix,
    raw_cols_to_sparse,
    sparse_to_dense,
    plot_matrix,
)
import sys, os, subprocess, shutil, logging
from docopt import docopt
import numpy as np


class AbstractCommand:
    """
    Base class for the commands
    """

    def __init__(self, command_args, global_args):
        """Initialize the commands"""
        self.args = docopt(self.__doc__, argv=command_args)
        self.global_args = global_args

    def execute(self):
        """Execute the commands"""
        raise NotImplementedError


class Iteralign(AbstractCommand):
    """
    Truncate reads from a fastq file to 20 basepairs and iteratively extend and
    re-align the unmapped reads to optimize the proportion of uniquely aligned
    reads in a 3C library.

    usage:
        iteralign [--minimap2] [--threads=1] [--min_len=20] [--tempdir DIR] --out_sam=FILE --fasta=FILE <reads.fq>

    arguments:
        reads.fq                Fastq file containing the reads to be aligned

    options:
        -f FILE, --fasta=FILE   Fasta file on which to map the reads.
        -t INT, --threads=INT  Number of parallel threads allocated for the alignment [default: 1].
        -T DIR, --tempdir=DIR  Temporary directory. Defaults to current directory.
        -m, --minimap2     If set, use minimap2 instead of bowtie2 for the alignment.
        -l INT, --min_len=INT  Length to which the reads should be truncated [default: 20].
        -o FILE, --out_sam=FILE Path where the alignment will be written in SAM format.
    """

    def execute(self):
        if not self.args["--tempdir"]:
            self.args["--tempdir"] = "."
        if not self.args["--minimap2"]:
            self.args["--minimap2"] = False
        temp_directory = generate_temp_dir(self.args["--tempdir"])
        iterative_align(
            self.args["<reads.fq>"],
            self.args["--tempdir"],
            self.args["--fasta"],
            self.args["--threads"],
            self.args["--out_sam"],
            self.args["--minimap2"],
        )
        # Deletes the temporary folder
        shutil.rmtree(temp_directory)


class Digest(AbstractCommand):
    """
    Digests a fasta file into fragments based on a restriction enzyme or a
    fixed chunk size. Generates two output files into the target directory
    named "info_contigs.txt" and "fragments_list.txt"

    usage:
        digest [--plot] [--figdir=FILE] [--circular] [--size=INT] [--outdir=DIR] --enzyme=ENZ <fasta>

    arguments:
        fasta                     Fasta file to be digested

    options:
        -c, --circular           Specify if the genome is circular.
        -e ENZ, --enzyme=ENZ     A restriction enzyme or an integer representing chunk sizes (in bp)
        -s INT, --size=INT       Minimum size threshold to keep fragments [default: 0]
        -o DIR, --outdir=DIR     Directory where the fragments and contigs files will be written. Defaults to current directory.
        -p, --plot               Show a histogram of fragment length distribution after digestion.
        -f FILE, --figdir=FILE   Path to the directory of the output figure. By default, the figure is only shown but not saved.

    output:
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes

    """

    def execute(self):
        # If circular is not specified, change it from None to False
        if not self.args["--circular"]:
            self.args["--circular"] = False
        if not self.args["--outdir"]:
            self.args["--outdir"] = os.getcwd()
        if not self.args["--figdir"]:
            self.args["--figdir"] = None
        write_frag_info(
            self.args["<fasta>"],
            self.args["--enzyme"],
            self.args["--size"],
            output_dir=self.args["--outdir"],
            circular=self.args["--circular"],
        )

        frag_len(
            output_dir=self.args["--outdir"],
            plot=self.args["--plot"],
            fig_path=os.path.join(self.args["--figdir"], "frags_hist.pdf"),
        )


class Filter(AbstractCommand):
    """
    Filters spurious 3C events such as loops and uncuts from the library based
    on a minimum distance threshold automatically estimated from the library by
    default. Can also plot 3C library statistics.

    usage:
        filter [--interactive | --thresholds INT,INT] [--plot] [--figdir FILE] <input> <output>

    arguments:
        input       2D BED file containing coordinates of Hi-C interacting pairs,
                    the index of their restriction fragment and their strands.
        output      Path to the filtered file, in the same format as the input.

    options:
        -i, --interactive                 Interactively shows plots and asks for thresholds.
        -t INT-INT, --thresholds=INT-INT  Manually defines integer values for the thresholds in the order [uncut, loop].
        -p, --plot                        Generates plots of library composition and 3C events abundance.
        -f DIR, --figdir=DIR              Path to the output figure directory. By default, the figure is only shown but not saved.
    """

    def execute(self):
        figpath = None
        output_handle = open(self.args["<output>"], "w")
        if self.args["--thresholds"]:
            # Thresholds supplied by user beforehand
            uncut_thr, loop_thr = self.args["--thresholds"].split("-")
            try:
                uncut_thr = int(uncut_thr)
                loop_thr = int(loop_thr)
            except ValueError:
                print("You must provide integer numbers for the thresholds.")
        else:
            # Threshold defined at runtime
            if self.args["--figdir"]:
                figpath = os.path.join(
                    self.args["--figdir"], "event_distance.pdf"
                )
            with open(self.args["<input>"]) as handle_in:
                uncut_thr, loop_thr = get_thresholds(
                    handle_in,
                    interactive=self.args["--interactive"],
                    plot_events=self.args["--plot"],
                    fig_path=figpath,
                )
        # Filter library and write to output file
        figpath = None
        if self.args["--figdir"]:
            figpath = os.path.join(
                self.args["--figdir"], "event_distribution.pdf"
            )
        with open(self.args["<input>"]) as handle_in:
            filter_events(
                handle_in,
                output_handle,
                uncut_thr,
                loop_thr,
                plot_events=self.args["--plot"],
                fig_path=figpath,
            )


class View(AbstractCommand):
    """
    Visualize a Hi-C matrix file as a heatmap of contact frequencies. Allows to
    tune visualisation by binning and normalizing the matrix, and to save the
    output image to disk. If no output is specified, the output is displayed.

    usage:
        view [--binning=1] [--frags FILE] [--normalize] [--max=99] [--output=IMG] <contact_map>

    arguments:
        contact_map             Sparse contact matrix in GRAAL format

    options:
        -b INT[bp|kb|Mb], --binning=INT[bp|kb|Mb]   Subsampling factor or fix value in basepairs to use for binning [default: 1].
        -f FILE, --frags FILE                       Required for bp binning. Tab-separated file with headers, containing fragments start position in the 3rd column, as generated by hicstuff pipeline.
        -n, --normalize                             Should SCN normalization be performed before rendering the matrix ?
        -m INT, --max=INT                           Saturation threshold. Maximum pixel value is set to this percentile [default: 99].
        -o IMG, --output=IMG                        Path where the matrix will be stored in PNG format.
    """

    def execute(self):

        input_map = self.args["<contact_map>"]
        bp_unit = False
        binsuffix = {"B": 1, "K": 1000, "M": 10e6, "G": 10e9}
        bin_str = self.args["--binning"].upper()
        try:
            # Subsample binning
            binning = int(bin_str)
        except ValueError:
            if re.match(r"^[0-9]+[KM]?B[P]?$", bin_str):
                # Extract unit and multiply accordingly for fixed bp binning
                unit_pos = re.search(r"[KM]?B[P]?$", bin_str).start()
                bp_unit = bin_str[unit_pos:]
                binning = int(bin_str[:unit_pos]) * binsuffix[bp_unit[0]]
                # Only keep 3rd column (start pos) and skip header
                if not self.args["--frags"]:
                    print(
                        "Error: A fragment file must be provided to perform "
                        "basepair binning. See hicstuff view --help",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                # Load positions from fragments list
                pos = np.genfromtxt(
                    self.args["--frags"],
                    delimiter="\t",
                    usecols=(2,),
                    skip_header=1,
                    dtype=np.int64,
                )
            else:
                print(
                    "Please provide an integer or basepair value for binning.",
                    file=sys.stderr,
                )
                raise

        vmax = float(self.args["--max"])
        output_file = self.args["--output"]
        raw_map = load_raw_matrix(input_map)
        sparse_map = raw_cols_to_sparse(raw_map)
        if self.args["--normalize"]:
            sparse_map = normalize_sparse(sparse_map, norm="SCN")

        if binning > 1:
            if bp_unit:
                binned_map, binned_frags = bin_bp_sparse(
                    M=sparse_map, positions=pos, bin_len=binning
                )
                pass
            else:
                binned_map = bin_sparse(
                    M=sparse_map, subsampling_factor=binning
                )
        else:
            binned_map = sparse_map

        try:
            dense_map = sparse_to_dense(binned_map)
            plot_matrix(dense_map, filename=output_file, vmax=vmax)
        except MemoryError:
            print("Contact map is too large to load, try binning more")


class Pipeline(AbstractCommand):
    """
    Entire Pipeline to process fastq files into a Hi-C matrix. Uses all the individual components of hicstuff.

    usage:
        pipeline [--quality_min=INT] [--duplicates] [--size=INT] [--no_cleanup]
                 [--threads=INT] [--minimap2] [--bedgraph] [--prefix=PREFIX]
                 [--tmpdir=DIR] [--iterative] [--outdir=DIR] [--filter]
                 [--enzyme=ENZ] [--plot] --fasta=FILE <fq1> <fq2>

    arguments:
        fq1:             Forward fastq file
        fq2:             Reverse fastq file

    options:
        -e ENZ, --enzyme=ENZ       Restriction enzyme if a string, or chunk size (i.e. resolution) if a number. [default: 5000]
        -f FILE, --fasta=FILE      Reference genome to map against in FASTA format
        -o DIR, --outdir=DIR       Output directory. Defaults to the current directory.
        -p, --plot                 Generates plots in the output directory at different steps of the pipeline.
        -P PREFIX, --prefix=PREFIX Overrides default GRAAL-compatible filenames and use a prefix with extensions instead.
        -q INT, --quality_min=INT  Minimum mapping quality for selecting contacts. [default: 30].
        -s INT, --size=INT         Minimum size threshold to consider contigs. Keep all contigs by default. [default: 0]
        -n, --no_cleanup           If enabled, intermediary BED files will be kept after generating the contact map. Disabled by defaut.
        -b, --bedgraph             If enabled, generates a sparse matrix in 2D Bedgraph format (cooler-compatible) instead of GRAAL-compatible format.
        -t INT, --threads=INT      Number of threads to allocate. [default: 1].
        -T DIR, --tmpdir=DIR          Directory for storing intermediary BED files and temporary sort files. Defaults to the output directory.
        -m, --minimap2             Use the minimap2 aligner instead of bowtie2. Not enabled by default.
        -i, --iterative            Map reads iteratively using hicstuff iteralign, by truncating reads to 20bp and then repeatedly extending and aligning them.
        -F, --filter               Filter out spurious 3C events (loops and uncuts) using hicstuff filter. Requires -e to be a restriction enzyme, not a chunk size.
        -C, --circular             Enable if the genome is circular.
        -d, --duplicates:          If enabled, trims (10bp) adapters and remove PCR duplicates prior to mapping. Only works if reads start with a 10bp sequence. Not enabled by default.
        -h, --help                 Display this help message.

    output:
        abs_fragments_contacts_weighted.txt: the sparse contact map
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes
    """

    def execute(self):

        if self.args["--filter"] and self.args["--enzyme"].isdigit():
            raise ValueError(
                "You cannot filter without specifying a restriction enzyme."
            )
        if not self.args["--outdir"]:
            self.args["--outdir"] = os.getcwd()

        str_args = " "
        # Pass formatted arguments to bash
        for arg, val in self.args.items():
            # Handle positional arguments individually
            if arg == "<fq1>":
                str_args += "-1 " + val
            elif arg == "<fq2>":
                str_args += "-2 " + val
            # Ignore value of flags (only add name)
            elif val is True:
                str_args += arg
            # Skip flags that are not specified
            elif val in (None, False):
                continue
            else:
                str_args += arg + " " + val
            str_args += " "
        subprocess.call("bash yahcp" + str_args, shell=True)
