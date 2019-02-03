# Structure based on Rémy Greinhofer (rgreinho) tutorial on subcommands in
# docopt : https://github.com/rgreinho/docopt-subcommands-example
# cmdoret, 20181412
from hicstuff.hicstuff import (
    bin_sparse,
    normalize_sparse,
    bin_bp_sparse,
    trim_sparse,
    despeckle_simple,
    scalogram,
    distance_law,
)
import re
from hicstuff.iteralign import *
from hicstuff.digest import write_frag_info, frag_len
from hicstuff.filter import get_thresholds, filter_events
from hicstuff.view import (
    load_raw_matrix,
    raw_cols_to_sparse,
    sparse_to_dense,
    plot_matrix,
)
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import sys, os, subprocess, shutil
from matplotlib import pyplot as plt
from docopt import docopt
import pandas as pd
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
        iteralign [--minimap2] [--threads=1] [--min_len=20] [--tempdir DIR]
                  --out_sam=FILE --fasta=FILE <reads.fq>

    arguments:
        reads.fq                Fastq file containing the reads to be aligned

    options:
        -f FILE, --fasta=FILE   Fasta file on which to map the reads.
        -t INT, --threads=INT   Number of parallel threads allocated for the
                                alignment [default: 1].
        -T DIR, --tempdir=DIR   Temporary directory. Defaults to current
                                directory.
        -m, --minimap2          If set, use minimap2 instead of bowtie2 for
                                the alignment.
        -l INT, --min_len=INT   Length to which the reads should be
                                truncated [default: 20].
        -o FILE, --out_sam=FILE Path where the alignment will be written in
                                SAM format.
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

    """

    def execute(self):
        # If circular is not specified, change it from None to False
        if not self.args["--circular"]:
            self.args["--circular"] = False
        if not self.args["--outdir"]:
            self.args["--outdir"] = os.getcwd()
        # Create output directory if it does not exist
        if not os.path.exists(self.args["--outdir"]):
            os.makedirs(self.args["--outdir"])
        if self.args["--figdir"]:
            figpath = os.path.join(self.args["--figdir"], "frags_hist.pdf")
        else:
            figpath = None
        # Split into a list if multiple enzymes given
        enzyme = self.args["--enzyme"]
        if re.search(r",", enzyme):
            enzyme = enzyme.split(",")

        write_frag_info(
            self.args["<fasta>"],
            enzyme,
            self.args["--size"],
            output_dir=self.args["--outdir"],
            circular=self.args["--circular"],
        )

        frag_len(
            output_dir=self.args["--outdir"], plot=self.args["--plot"], fig_path=figpath
        )


class Filter(AbstractCommand):
    """
    Filters spurious 3C events such as loops and uncuts from the library based
    on a minimum distance threshold automatically estimated from the library by
    default. Can also plot 3C library statistics.

    usage:
        filter [--interactive | --thresholds INT-INT] [--plot]
               [--figdir FILE] [--prefix STR] <input> <output>

    arguments:
        input       2D BED file containing coordinates of Hi-C interacting
                    pairs, the index of their restriction fragment and their
                    strands.
        output      Path to the filtered file, in the same format as the input.

    options:
        -f DIR, --figdir=DIR              Path to the output figure directory.
                                          By default, the figure is only shown
                                          but not saved.
        -i, --interactive                 Interactively shows plots and asks
                                          for thresholds.
        -p, --plot                        Shows plots of library composition
                                          and 3C events abundance.
        -P STR, --prefix STR              If the library has a name, it will
                                          be shown on the figures.
        -t INT-INT, --thresholds=INT-INT  Manually defines integer values for
                                          the thresholds in the order
                                          [uncut, loop].
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
                figpath = os.path.join(self.args["--figdir"], "event_distance.pdf")
                if not os.path.exists(self.args["--figdir"]):
                    os.makedirs(self.args["--figdir"])
            with open(self.args["<input>"]) as handle_in:
                uncut_thr, loop_thr = get_thresholds(
                    handle_in,
                    interactive=self.args["--interactive"],
                    plot_events=self.args["--plot"],
                    fig_path=figpath,
                    prefix=self.args["--prefix"],
                )
        # Filter library and write to output file
        figpath = None
        if self.args["--figdir"]:
            figpath = os.path.join(self.args["--figdir"], "event_distribution.pdf")

        with open(self.args["<input>"]) as handle_in:
            filter_events(
                handle_in,
                output_handle,
                uncut_thr,
                loop_thr,
                plot_events=self.args["--plot"],
                fig_path=figpath,
                prefix=self.args["--prefix"],
            )


class View(AbstractCommand):
    """
    Visualize a Hi-C matrix file as a heatmap of contact frequencies. Allows to
    tune visualisation by binning and normalizing the matrix, and to save the
    output image to disk. If no output is specified, the output is displayed.

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
    """

    def process_matrix(self, sparse_map):
        """
        Performs any combination of binning, normalisation, log transformation,
        trimming and subsetting based on the attributes of the instance class.
        """
        ### BINNING ###
        if self.binning > 1:
            if self.bp_unit:
                binned_map, binned_frags = bin_bp_sparse(
                    M=sparse_map, positions=self.pos, bin_len=self.binning
                )

            else:
                binned_map = bin_sparse(M=sparse_map, subsampling_factor=self.binning)
        else:
            binned_map = sparse_map

        ### NORMALIZATION ###
        if self.args["--normalize"]:
            binned_map = normalize_sparse(binned_map, norm="SCN")

        ### LOG VALUES ###
        if self.args["--log"]:
            binned_map = binned_map.log1p()

        ### ZOOM REGION ###
        if self.args["--region"]:
            if not self.args["--frags"]:
                print(
                    "Error: A fragment file must be provided to subset "
                    "genomic regions. See hicstuff view --help",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Load positions from fragments list
            reg_pos = pd.read_csv(self.args["--frags"], delimiter="\t", usecols=(1, 2))
            # Readjust bin coords post binning
            if self.binning:
                if self.bp_unit:
                    binned_start = np.append(
                        np.where(binned_frags == 0)[0], binned_frags.shape[0]
                    )
                    num_binned = binned_start[1:] - binned_start[:-1]
                    chr_names = np.unique(reg_pos.iloc[:, 0])
                    binned_chrom = np.repeat(chr_names, num_binned)
                    reg_pos = pd.DataFrame({0: binned_chrom, 1: binned_frags[:, 0]})
                else:
                    reg_pos = reg_pos.iloc[:: self.binning, :]

            region = self.args["--region"]
            if ";" in region:
                reg1, reg2 = region.split(";")
                reg1 = parse_ucsc(reg1, reg_pos)
                reg2 = parse_ucsc(reg2, reg_pos)
            else:
                region = parse_ucsc(region, reg_pos)
                reg1 = reg2 = region
            binned_map = binned_map.tocsr()
            binned_map = binned_map[reg1[0] : reg1[1], reg2[0] : reg2[1]]
            binned_map = binned_map.tocoo()

        ### TRIMMING ###
        if self.args["--trim"]:
            try:
                trim_std = float(self.args["--trim"])
            except ValueError:
                print(
                    "You must specify a number of standard deviations for " "trimming"
                )
                raise
            binned_map = trim_sparse(binned_map, n_std=trim_std)

        return binned_map

    def execute(self):

        input_map = self.args["<contact_map>"]
        cmap = self.args["--cmap"]
        self.bp_unit = False
        bin_str = self.args["--binning"].upper()
        try:
            # Subsample binning
            self.binning = int(bin_str)
        except ValueError:
            if re.match(r"^[0-9]+[KMG]?B[P]?$", bin_str):
                if not self.args["--frags"]:
                    print(
                        "Error: A fragment file must be provided to perform "
                        "basepair binning. See hicstuff view --help",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                # Load positions from fragments list
                self.pos = np.genfromtxt(
                    self.args["--frags"],
                    delimiter="\t",
                    usecols=(2,),
                    skip_header=1,
                    dtype=np.int64,
                )
                self.binning = parse_bin_str(bin_str)
                self.bp_unit = True
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
        processed_map = self.process_matrix(sparse_map)
        # If 2 matrices given compute log ratio
        if self.args["<contact_map2>"]:
            raw_map2 = load_raw_matrix(self.args["<contact_map2>"])
            sparse_map2 = raw_cols_to_sparse(raw_map2)
            processed_map2 = self.process_matrix(sparse_map2)
            if sparse_map2.shape != sparse_map.shape:
                print(
                    "Error: You cannot compute the ratio of matrices with "
                    "different dimensions",
                    file=sys.stderr,
                )
            # Get log of values for both maps
            processed_map.data = np.log2(processed_map.data)
            processed_map2.data = np.log2(processed_map2.data)
            # Note: Taking diff of logs instead of log of ratio because sparse
            # mat division yields dense matrix in current implementation.
            # Changing base to 2 afterwards.
            processed_map = processed_map.tocsr() - processed_map2.tocsr()
            processed_map = processed_map.tocoo()
            cmap = "coolwarm"

        vmax = np.percentile(processed_map.data, vmax)
        try:
            dense_map = sparse_to_dense(processed_map)
            if self.args["--despeckle"]:
                dense_map = despeckle_local(dense_map)
            vmin = 0
            if self.args["<contact_map2>"]:
                vmin, vmax = -2, 2
            plot_matrix(
                dense_map, filename=output_file, vmin=vmin, vmax=vmax, cmap=cmap
            )
        except memoryerror:
            print("contact map is too large to load, try binning more")


class Pipeline(AbstractCommand):
    """
    Entire Pipeline to process fastq files into a Hi-C matrix. Uses all the
    individual components of hicstuff.

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
    """

    def execute(self):
        if self.args["--pairs"] or self.args["--sam"]:
            # If starting from middle of pipeline, do not remove intermediary
            # files to prevent deleting input.
            self.args["--no-cleanup"] = True

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
            if arg in {"<fq1>", "<sam1>", "<bed2D>"} and val:
                str_args += "-1 " + val
            elif arg in {"<fq2>", "<sam2>"} and val:
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
        # Set the pipeline to start from later step if specified
        if self.args["--pairs"]:
            str_args += "-S 3"
        elif self.args["--sam"]:
            str_args += "-S 2"
        subprocess.call("bash yahcp" + str_args, shell=True)


class Plot(AbstractCommand):
    """
    Generate the specified type of plot.

    usage:
        plot [--cmap=NAME] [--range=INT-INT] [--coord=INT-INT] [--threads=INT]
             [--output=FILE] [--max=INT] [--centro] [--process] [--despeckle]
             (--scalogram | --distance_law) <contact_map>

    argument:
        <contact_map> The sparse Hi-C contact matrix.

    options:
        -c INT-INT, --coord INT-INT        The bins of the matrix to use for
                                           the plot (e.g. coordinates of a
                                           single chromosome).
        -C NAME, --cmap NAME               The matplotlib colormap to use for
                                           the plot. [default: viridis]
        -d, --despeckle                    Remove speckles (small intense spots)
                                           from the matrix.
        -f FILE, --frags FILE              The path to the hicstuff fragments
                                           file.
        -l, --distance_law                 Plot the distance law of the matrix.
        -m INT, --max INT                  Saturation threshold in percentile
                                           of pixel values. [default: 99]
        -o FILE, --output FILE             Output file where the plot should be
                                           saved. Plot is only displayed by
                                           default.
        -p, --process                      Process the matrix first (trim,
                                           normalize)
        -r INT-INT, --range INT-INT        The range of contact distance to
                                           look at. No limit by default.
        -s, --scalogram                    Plot the scalogram of the contact map.
        -t INT, --threads=Inter            Number of processes to run in
                                           parallel for despeckling. [default: 1]
    """

    def execute(self):
        try:
            if self.args["--range"]:
                lower, upper = self.args["--range"].split("-")
                lower = int(lower)
                upper = int(upper)
            if self.args["--coord"]:
                start, end = self.args["--coord"].split("-")
                start = int(start)
                end = int(end)
        except ValueError:
            print(
                "Range must be provided using two integers separated by '-'.",
                "E.g: 1-100.",
            )
        input_map = self.args["<contact_map>"]

        vmax = float(self.args["--max"])
        output_file = self.args["--output"]
        S = load_raw_matrix(input_map)
        S = raw_cols_to_sparse(S)
        S = csr_matrix(S)
        if not self.args["--range"]:
            lower = 0
            upper = S.shape[0]

        if self.args["--process"]:
            S = trim_sparse(S, n_std=3)
            S = normalize_sparse(S, norm="SCN")
        if self.args["--despeckle"]:
            S = despeckle_simple(S, threads=self.args["--threads"])

        if self.args["--coord"]:
            S = S[start:end, start:end]
        if self.args["--scalogram"]:
            D = S.todense()
            D = np.fliplr(np.rot90(scalogram(D), k=-1))
            plt.contourf(D[:, lower:upper], cmap=self.args["--cmap"])
        elif self.args["--distance_law"]:
            subp, tmpidx = distance_law(S, log_bins=True)
            plt.plot(tmpidx[2:], subp[2:])
            plt.xscale("log")
            plt.yscale("log")
        if self.args["--output"]:
            plt.savefig(self.args["--output"])
        else:
            plt.show()


def parse_bin_str(bin_str):
    """
    Take a basepair binning string as input and converts it into
    corresponding basepair values.
    Parameters
    ----------
    bin_str : str
        A basepair region (e.g. 150KB). Unit can be BP, KB, MB, GB.
    Returns
    -------
    int :
        The number of basepair corresponding to the binning string.
    """
    bin_str = bin_str.upper()
    binsuffix = {"B": 1, "K": 1000, "M": 10e6, "G": 10e9}
    unit_pos = re.search(r"[KMG]?B[P]?$", bin_str).start()
    bp_unit = bin_str[unit_pos:]
    # Extract unit and multiply accordingly for fixed bp binning
    binning = int(bin_str[:unit_pos]) * binsuffix[bp_unit[0]]

    return binning


def parse_ucsc(ucsc_str, bins):
    """
    Take a UCSC region in UCSC notation and a list of bin chromosomes and
    positions (in basepair) and converts it to range of bins.
    Parameters
    ----------
    ucsc_str : str
        The region string in UCSC notation (e.g. chr1:1000-2000)
    bins : pandas.DataFrame
        Dataframe of two columns containing the chromosome and start
        position of each bin. Each row must be one bin.
    Returns
    -------
    tuple :
        A tuple containing the bin range containing in the requested region.

    """
    chrom, bp = ucsc_str.split(":")
    bp = bp.replace(",", "").upper()
    start, end = bp.split("-")
    try:
        start, end = int(start), int(end)
    except ValueError:
        start, end = parse_bin_str(start), parse_bin_str(end)
    bins["id"] = bins.index
    chrombins = bins.loc[bins.iloc[:, 0] == chrom, :]
    start = max([start, 1])
    start = max(chrombins.id[chrombins.iloc[:, 1] // start == 0])
    end = max(chrombins.id[chrombins.iloc[:, 1] // end == 0])
    return (start, end)
