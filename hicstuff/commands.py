#!/usr/bin/env python3
# coding: utf-8

"""Abstract command classes for hicstuff

This module contains all classes related to hicstuff
commands:

    -iteralign (iterative mapping)
    -digest (genome chunking)
    -filter (Hi-C 'event' sorting: loops, uncuts, weird
     and 'true contacts')
    -view (map visualization)
    -pipeline (whole contact map generation)
    -distancelaw (Analysis tool and plot for the distance law)

Running 'pipeline' implies running 'digest', but not
iteralign or filter unless specified, because they can
take up a lot of time for dimnishing returns.

Note
----
Structure based on Rémy Greinhofer (rgreinho) tutorial on subcommands in
docopt : https://github.com/rgreinho/docopt-subcommands-example
cmdoret, 20181412

Raises
------
NotImplementedError
    Will be raised if AbstractCommand is called for
    some reason instead of one of its children.
ValueError
    Will be raised if an incorrect chunking method (e.g.
    not an enzyme or number or invalid range view is
    specified.
"""
import re
import hicstuff.view as hcv
import hicstuff.hicstuff as hcs
import hicstuff.digest as hcd
import hicstuff.iteralign as hci
import hicstuff.filter as hcf
import hicstuff.io as hio
from hicstuff.log import logger
import hicstuff.pipeline as hpi
import hicstuff.distance_law as hcdl
from scipy.sparse import csr_matrix
import sys, os, shutil, csv
from os.path import join, basename
from matplotlib import pyplot as plt
from matplotlib import cm
from docopt import docopt
import pandas as pd
import numpy as np


class AbstractCommand:
    """Abstract base command class

    Base class for the commands from which
    other hicstuff commands derive.
    """

    def __init__(self, command_args, global_args):
        """Initialize the commands"""
        self.args = docopt(self.__doc__, argv=command_args)
        self.global_args = global_args

    def execute(self):
        """Execute the commands"""
        raise NotImplementedError


class Iteralign(AbstractCommand):
    """Iterative mapping command

    Truncate reads from a fastq file to 20 basepairs and iteratively extend and
    re-align the unmapped reads to optimize the proportion of uniquely aligned
    reads in a 3C library.

    usage:
        iteralign [--aligner=bowtie2] [--threads=1] [--min_len=20]
                  [--tempdir DIR] --out_sam=FILE --genome=FILE <reads.fq>

    arguments:
        reads.fq                Fastq file containing the reads to be aligned

    options:
        -g, --genome=FILE        The genome on which to map the reads. Must be
                                 The path to the bowtie2 index if using bowtie2
                                 or to the genome in fasta format if using
                                 minimap2.
        -t, --threads=INT        Number of parallel threads allocated for the
                                 alignment [default: 1].
        -T, --tempdir=DIR        Temporary directory. Defaults to current
                                 directory.
        -a, --aligner=bowtie2    Choose alignment software between bowtie2 and
                                 minimap2. [default: bowtie2]
        -l, --min_len=INT        Length to which the reads should be
                                 truncated [default: 20].
        -o, --out_sam=FILE       Path where the alignment will be written in
                                 SAM format.
    """

    def execute(self):
        if not self.args["--tempdir"]:
            self.args["--tempdir"] = "."
        temp_directory = hio.generate_temp_dir(self.args["--tempdir"])
        hci.iterative_align(
            self.args["<reads.fq>"],
            temp_directory,
            self.args["--genome"],
            self.args["--threads"],
            self.args["--out_sam"],
            aligner=self.args["--aligner"],
            min_len=int(self.args["--min_len"]),
        )
        # Deletes the temporary folder
        shutil.rmtree(temp_directory)


class Digest(AbstractCommand):
    """Genome chunking command

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
        -s, --size=INT                  Minimum size threshold to keep
                                        fragments. [default: 0]
        -o, --outdir=DIR                Directory where the fragments and
                                        contigs files will be written.
                                        Defaults to current directory.
        -p, --plot                      Show a histogram of fragment length
                                        distribution after digestion.
        -f, --figdir=FILE               Path to directory of the output figure.
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
            figpath = join(self.args["--figdir"], "frags_hist.pdf")
        else:
            figpath = None
        # Split into a list if multiple enzymes given
        enzyme = self.args["--enzyme"]
        if re.search(r",", enzyme):
            enzyme = enzyme.split(",")

        hcd.write_frag_info(
            self.args["<fasta>"],
            enzyme,
            self.args["--size"],
            output_dir=self.args["--outdir"],
            circular=self.args["--circular"],
        )

        hcd.frag_len(
            output_dir=self.args["--outdir"], plot=self.args["--plot"], fig_path=figpath
        )


class Filter(AbstractCommand):
    """Mapping event filtering command

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
        -f, --figdir=DIR                  Path to the output figure directory.
                                          By default, the figure is only shown
                                          but not saved.
        -i, --interactive                 Interactively shows plots and asks
                                          for thresholds.
        -p, --plot                        Shows plots of library composition
                                          and 3C events abundance.
        -P, --prefix STR                  If the library has a name, it will
                                          be shown on the figures.
        -t, --thresholds=INT-INT          Manually defines integer values for
                                          the thresholds in the order
                                          [uncut, loop]. Reads above those values
                                          are kept.
    """

    def execute(self):
        figpath = None
        if self.args["--thresholds"]:
            # Thresholds supplied by user beforehand
            uncut_thr, loop_thr = self.args["--thresholds"].split("-")
            try:
                uncut_thr = int(uncut_thr)
                loop_thr = int(loop_thr)
            except ValueError:
                logger.error("You must provide integer numbers for the thresholds.")
        else:
            # Threshold defined at runtime
            if self.args["--figdir"]:
                figpath = join(self.args["--figdir"], "event_distance.pdf")
                if not os.path.exists(self.args["--figdir"]):
                    os.makedirs(self.args["--figdir"])
            uncut_thr, loop_thr = hcf.get_thresholds(
                self.args["<input>"],
                interactive=self.args["--interactive"],
                plot_events=self.args["--plot"],
                fig_path=figpath,
                prefix=self.args["--prefix"],
            )
        # Filter library and write to output file
        figpath = None
        if self.args["--figdir"]:
            figpath = join(self.args["--figdir"], "event_distribution.pdf")

        hcf.filter_events(
            self.args["<input>"],
            self.args["<output>"],
            uncut_thr,
            loop_thr,
            plot_events=self.args["--plot"],
            fig_path=figpath,
            prefix=self.args["--prefix"],
        )


class View(AbstractCommand):
    """Contact map visualization command

    Visualize a Hi-C matrix file as a heatmap of contact frequencies. Allows to
    tune visualisation by binning and normalizing the matrix, and to save the
    output image to disk. If no output is specified, the output is displayed.

    usage:
        view [--binning=1] [--despeckle] [--frags FILE] [--trim INT]
             [--normalize] [--max=99] [--output=IMG] [--cmap=CMAP]
             [--log] [--circular] [--region=STR] <contact_map> [<contact_map2>]

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
        -f, --frags=FILE                 Required for bp binning. Tab-separated
                                         file with headers, containing
                                         fragments start position in the 3rd
                                         column, as generated by hicstuff
                                         pipeline.
        -l, --log                        Log transform pixel values to improve
                                         visibility of long range signals.
        -m, --max=INT                    Saturation threshold. Maximum pixel
                                         value is set to this percentile
                                         [default: 99].
        -n, --normalize                  Should SCN normalization be performed
                                         before rendering the matrix ?
        -o, --output=IMG                 Name of the image file where the view is stored.
        -r, --region=STR[;STR]           Only view a region of the contact map.
                                         Regions are specified as UCSC strings.
                                         (e.g.:chr1:1000-12000). If only one
                                         region is given, it is viewed on the
                                         diagonal. If two regions are given,
                                         The contacts between both are shown.
        -t, --trim=INT                   Trims outlier rows/columns from the
                                         matrix if the sum of their contacts
                                         deviates from the mean by more than
                                         INT standard deviations.
    """

    def process_matrix(self, sparse_map):
        """
        Performs any combination of binning, normalisation, log transformation,
        trimming and subsetting based on the attributes of the instance class.
        """
        # BINNING
        if self.binning > 1:
            if self.bp_unit:
                binned_map, binned_frags = hcs.bin_bp_sparse(
                    M=sparse_map, positions=self.pos, bin_len=self.binning
                )

            else:
                binned_map = hcs.bin_sparse(
                    M=sparse_map, subsampling_factor=self.binning
                )
        else:
            binned_map = sparse_map

        # NORMALIZATION
        if self.args["--normalize"]:
            binned_map = hcs.normalize_sparse(binned_map, norm="SCN")

        # LOG VALUES
        if self.args["--log"]:
            binned_map = binned_map.log1p()

        self.vmax = np.percentile(binned_map.data, self.vmax)
        # ZOOM REGION
        if self.args["--region"]:
            if not self.args["--frags"]:
                logger.error(
                    "A fragment file must be provided to subset "
                    "genomic regions. See hicstuff view --help"
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
                # 2 input regions: zoom anywhere in matrix
                self.symmetric = False
                reg1, reg2 = region.split(";")
                reg1 = parse_ucsc(reg1, reg_pos)
                reg2 = parse_ucsc(reg2, reg_pos)
            else:
                # Only 1 input region: zoom on diagonal
                region = parse_ucsc(region, reg_pos)
                reg1 = reg2 = region
            binned_map = binned_map.tocsr()
            binned_map = binned_map[reg1[0] : reg1[1], reg2[0] : reg2[1]]
            binned_map = binned_map.tocoo()

        # TRIMMING
        if self.args["--trim"]:
            try:
                trim_std = float(self.args["--trim"])
            except ValueError:
                logger.error(
                    "You must specify a number of standard deviations for " "trimming"
                )
                raise
            binned_map = hcs.trim_sparse(binned_map, n_std=trim_std)

        return binned_map

    def execute(self):

        input_map = self.args["<contact_map>"]
        cmap = self.args["--cmap"]
        self.vmax = float(self.args["--max"])
        self.bp_unit = False
        bin_str = self.args["--binning"].upper()
        self.symmetric = True
        try:
            # Subsample binning
            self.binning = int(bin_str)
        except ValueError:
            if re.match(r"^[0-9]+[KMG]?B[P]?$", bin_str):
                if not self.args["--frags"]:
                    logger.error(
                        "A fragment file must be provided to perform "
                        "basepair binning. See hicstuff view --help"
                    )
                    sys.exit(1)
                # Load positions from fragments list
                self.pos = hio.load_pos_col(self.args["--frags"], 2)
                self.binning = parse_bin_str(bin_str)
                self.bp_unit = True
            else:
                logger.error("Please provide an integer or basepair value for binning.")
                raise

        output_file = self.args["--output"]
        sparse_map = hio.load_sparse_matrix(input_map)
        processed_map = self.process_matrix(sparse_map)
        # If 2 matrices given compute log ratio
        if self.args["<contact_map2>"]:
            sparse_map2 = hio.load_sparse_matrix(self.args["<contact_map2>"])
            processed_map2 = self.process_matrix(sparse_map2)
            if sparse_map2.shape != sparse_map.shape:
                logger.error(
                    "You cannot compute the ratio of matrices with "
                    "different dimensions"
                )
            # Get log of values for both maps
            processed_map.data = np.log2(processed_map.data)
            processed_map2.data = np.log2(processed_map2.data)
            # Note: Taking diff of logs instead of log of ratio because sparse
            # mat division yields dense matrix in current implementation.
            # Changing base to 2 afterwards.
            processed_map = processed_map.tocsr() - processed_map2.tocsr()
            processed_map = processed_map.tocoo()
            processed_map.data[np.isnan(processed_map.data)] = 0.0
            cmap = "coolwarm"

        if self.args["--despeckle"]:
            processed_map = hcs.despeckle_simple(processed_map)
        try:
            if self.symmetric:
                dense_map = hcv.sparse_to_dense(processed_map, remove_diag=False)
            else:
                dense_map = processed_map.todense()
            self.vmin = 0
            if self.args["<contact_map2>"]:
                self.vmin, self.vmax = -2, 2
            hcv.plot_matrix(
                dense_map,
                filename=output_file,
                vmin=self.vmin,
                vmax=self.vmax,
                cmap=cmap,
            )
        except MemoryError:
            logger.error("contact map is too large to load, try binning more")


class Pipeline(AbstractCommand):
    """Whole (end-to-end) contact map generation command

    Entire Pipeline to process fastq files into a Hi-C matrix. Uses all the
    individual components of hicstuff.

    usage:
        pipeline [--quality-min=INT] [--size=INT] [--no-cleanup] [--start-stage=STAGE]
                 [--threads=INT] [--aligner=bowtie2] [--matfmt=FMT] [--prefix=PREFIX]
                 [--tmpdir=DIR] [--iterative] [--outdir=DIR] [--filter] [--enzyme=ENZ]
                 [--plot] [--circular] [--distance_law] [--duplicates]
                 [--centromeres=FILE] --genome=FILE <input1> [<input2>]

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
                                      Discordant with the centromeres option.   
        -e, --enzyme=ENZ              Restriction enzyme if a string, or chunk
                                      size (i.e. resolution) if a number. Can
                                      also be multiple comma-separated enzymes.
                                      [default: 5000]
        -g, --genome=FILE             Reference genome to map against. Path to
                                      the bowtie2 index if using bowtie2, or to
                                      a FASTA file if using minimap2.
        -F, --filter                  Filter out spurious 3C events (loops and
                                      uncuts) using hicstuff filter. Requires
                                      "-e" to be a restriction enzyme, not a
                                      chunk size.
        -S, --start-stage=STAGE       Define the starting point of the pipeline
                                      to skip some steps. Default is "fastq" to
                                      run from the start. Can also be "sam" to
                                      skip the alignment, "pairs" to start from a
                                      single pairs file or "pairs_idx" to skip
                                      fragment attribution and only build the 
                                      matrix. [default: fastq]
        -i, --iterative               Map reads iteratively using hicstuff
                                      iteralign, by truncating reads to 20bp
                                      and then repeatedly extending and
                                      aligning them.
        -a, --aligner=bowtie2         Alignment software to use. Can be either
                                      bowtie2 or minmap2. [default: bowtie2]
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
        -d, --distance-law            If enabled, generates a distance law file
                                      with the values of the probabilities to 
                                      have a contact between two distances for
                                      each chromosomes or arms if the file with
                                      the positions has been given. The values
                                      are not normalized, or averaged.
        -D, --duplicates              Filter out PCR duplicates based on read
                                      positions.
        -c, --centromeres=FILE        Positions of the centromeres separated by
                                      a space and in the same order than the 
                                      chromosomes. Discordant with the circular
                                      option.           

    output:
        abs_fragments_contacts_weighted.txt: the sparse contact map
        fragments_list.txt: information about restriction fragments (or chunks)
        info_contigs.txt: information about contigs or chromosomes
        hicstuff.log: details and statistics about the run.
    """

    def execute(self):

        if self.args["--filter"] and self.args["--enzyme"].isdigit():
            raise ValueError(
                "You cannot filter without specifying a restriction enzyme."
            )
        if not self.args["--outdir"]:
            self.args["--outdir"] = os.getcwd()
        if self.args["--matfmt"] not in ("GRAAL", "cooler"):
            raise ValueError("matfmt must be either cooler or GRAAL.")

        hpi.full_pipeline(
            genome=self.args["--genome"],
            input1=self.args["<input1>"],
            input2=self.args["<input2>"],
            enzyme=self.args["--enzyme"],
            circular=self.args["--circular"],
            out_dir=self.args["--outdir"],
            tmp_dir=self.args["--tmpdir"],
            plot=self.args["--plot"],
            min_qual=int(self.args["--quality-min"]),
            min_size=int(self.args["--size"]),
            threads=int(self.args["--threads"]),
            no_cleanup=self.args["--no-cleanup"],
            iterative=self.args["--iterative"],
            filter_events=self.args["--filter"],
            prefix=self.args["--prefix"],
            start_stage=self.args["--start-stage"],
            mat_fmt=self.args["--matfmt"],
            aligner=self.args["--aligner"],
            pcr_duplicates=self.args["--duplicates"],
            distance_law=self.args["--distance_law"],
            centromeres=self.args["--centromeres"],
        )


class Scalogram(AbstractCommand):
    """
    Generate a scalogram.

    usage:
        plot [--cmap=NAME] [--centromeres=FILE] [--frags=FILE] [--range=INT-INT]
             [--threads=INT] [--output=FILE] [--max=INT] [--process]
             [--indices=INT-INT] [--despeckle] <contact_map>

    argument:
        <contact_map> The sparse Hi-C contact matrix.

    options:
        -C, --cmap=NAME                    The matplotlib colormap to use for
                                           the plot. [default: viridis]
        -d, --despeckle                    Remove speckles (artifactual spots)
                                           from the matrix.
        -i, --indices=INT-INT              The bins of the matrix to use for
                                           the plot (e.g. coordinates of a
                                           single chromosome).
        -m, --max=INT                      Saturation threshold in percentile
                                           of pixel values. [default: 99]
        -o, --output=FILE                  Output file where the plot should be
                                           saved. Plot is only displayed by
                                           default.
        -p, --process                      Process the matrix first (trim,
                                           normalize)
        -r, --range=INT-INT                The range of contact distance to look
                                           at. No limit by default. Values in
                                           basepairs by default but a unit can
                                           be specified (kb, Mb, ...).
        -t, --threads=INT                  Parallel processes to run in for
                                           despeckling. [default: 1]
    """

    def execute(self):
        try:
            if self.args["--range"]:
                lower, upper = self.args["--range"].split("-")
                try:
                    lower, upper = int(lower), int(upper)
                except ValueError:
                    lower, upper = parse_bin_str(lower), parse_bin_str(upper)
            if self.args["--indices"]:
                start, end = self.args["--indices"].split("-")
                start = int(start)
                end = int(end)
        except ValueError:
            raise ValueError(
                "Range must be provided using two integers separated by '-'.",
                "E.g: 1-100.",
            )
        input_map = self.args["<contact_map>"]
        vmax = float(self.args["--max"])
        output_file = self.args["--output"]
        S = hio.load_sparse_matrix(input_map)
        good_bins = np.array(range(S.shape[0]))
        S = csr_matrix(S)
        if not self.args["--range"]:
            lower = 0
            upper = S.shape[0]

        if self.args["--process"]:
            good_bins = np.where(hcs.get_good_bins(S, n_std=3) == 1)[0]
            S = hcs.normalize_sparse(S, norm="SCN")
        if self.args["--despeckle"]:
            S = hcs.despeckle_simple(S, threads=self.args["--threads"])

        if self.args["--indices"]:
            S = S[start:end, start:end]
        D = hcv.sparse_to_dense(S)
        D = np.fliplr(np.rot90(hcs.scalogram(D), k=-1))
        plt.contourf(D[:, lower:upper], cmap=self.args["--cmap"])
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()


class Rebin(AbstractCommand):
    """
    Rebins a Hi-C matrix and modifies its fragment and chrom files accordingly.
    Output files are given the same name as the input files, in the target
    directory.
    usage:
        rebin [--binning=1] --frags=FILE --chrom=FILE --outdir=DIR
               <contact_map>

    arguments:
        contact_map             Sparse contact matrix in GRAAL format

    options:
        -b, --binning=INT[bp|kb|Mb|Gb]   Subsampling factor or fix value in
                                         basepairs to use for binning
                                         [default: 1].
        -f, --frags=FILE                 Tab-separated file with headers,
                                         containing fragments start position in
                                         the 3rd column, as generated by
                                         hicstuff pipeline.
        -c, --chrom=file                 Tab-separated with headers, containing
                                         chromosome names, size, number of
                                         restriction fragments.
        -o, --outdir=DIR                 Directory where the new binned files
                                         will be written.
    """

    def execute(self):
        bin_str = self.args["--binning"].upper()
        # Load positions from fragments list and chromosomes from chrom file
        frags = pd.read_csv(self.args["--frags"], sep="\t")
        chromlist = pd.read_csv(self.args["--chrom"], sep="\t")
        outdir = self.args["--outdir"]
        # Create output directory if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        bp_unit = False
        try:
            # Subsample binning
            binning = int(bin_str)
        except ValueError:
            # Basepair binning
            if re.match(r"^[0-9]+[KMG]?B[P]?$", bin_str):
                if not self.args["--frags"]:
                    logger.error(
                        "Error: A fragment file must be provided to perform "
                        "basepair binning. See hicstuff rebin --help"
                    )
                    sys.exit(1)
                binning = parse_bin_str(bin_str)
                bp_unit = True
            else:
                logger.error("Please provide an integer or basepair value for binning.")
                raise
        map_path = self.args["<contact_map>"]
        hic_map = hio.load_sparse_matrix(map_path)
        chromnames = np.unique(frags.chrom)
        if bp_unit:
            # Basepair binning
            hic_map, _ = hcs.bin_bp_sparse(hic_map, frags.start_pos, binning)
            for chrom in chromnames:
                # For all chromosomes, get new bin start positions
                bin_id = frags.loc[frags.chrom == chrom, "start_pos"] // binning
                frags.loc[frags.chrom == chrom, "id"] = bin_id + 1
                frags.loc[frags.chrom == chrom, "start_pos"] = binning * bin_id
                bin_ends = binning * bin_id + binning
                # Do not allow bin ends to be larger than chrom size
                chromsize = chromlist.length[chromlist.contig == chrom].values[0]
                # bin_ends.iloc[-1] = min([bin_ends.iloc[-1], chromsize])
                bin_ends[bin_ends > chromsize] = chromsize
                frags.loc[frags.chrom == chrom, "end_pos"] = bin_ends
                frags.loc[frags.chrom == chrom, "end_pos"] = bin_ends
                frags.loc[frags.chrom == chrom, "end_pos"] = bin_ends

        else:
            # Subsample binning
            hic_map = hcs.bin_sparse(hic_map, binning)
            # Use index for binning, but keep 1-indexed.
            # Exception when binning is 1 (no binning) where no need to shift
            shift_id = 0 if binning == 1 else 1
            frags.id = (frags.id // binning) + shift_id
        # Save original columns order
        col_ordered = list(frags.columns)
        # Get new start and end position for each bin
        frags = frags.groupby(["chrom", "id"], sort=False)
        positions = frags.agg({"start_pos": "min", "end_pos": "max"})
        positions.reset_index(inplace=True)
        # Compute mean for all added features in each index bin
        # Normally only other feature is GC content
        features = frags.agg("mean")
        features.reset_index(inplace=True)
        # set new bins positions
        frags = features
        frags.loc[:, positions.columns] = positions
        frags["size"] = frags.end_pos - frags.start_pos
        cumul_bins = 0
        for chrom in chromnames:
            n_bins = frags.start_pos[frags.chrom == chrom].shape[0]
            chromlist.loc[chromlist.contig == chrom, "n_frags"] = n_bins
            chromlist.loc[chromlist.contig == chrom, "cumul_length"] = cumul_bins
            cumul_bins += n_bins

        # Write 3 binned output files
        hio.save_sparse_matrix(hic_map, join(outdir, basename(map_path)))
        # Keep original column order
        frags = frags.reindex(columns=col_ordered)
        frags.to_csv(
            join(outdir, basename(self.args["--frags"])), index=False, sep="\t"
        )
        chromlist.to_csv(
            join(outdir, basename(self.args["--chrom"])), index=False, sep="\t"
        )


class Subsample(AbstractCommand):
    """
    Subsample contacts from a Hi-C matrix. Probability of sampling is proportional
    to the intensity of the bin.
    usage:
        subsample  [--prop=FLOAT] <contact_map> <subsampled_map>

    arguments:
        contact_map             Sparse contact matrix in GRAAL format
        subsampled_map          Output map containing only a fraction of the
                                contacts

    options:
        -p, --prop=FLOAT                 Proportion of contacts to sample from
                                         the input matrix. Given as a value
                                         between 0 and 1. [default: 0.1]
    """

    def execute(self):
        map_in = hio.load_sparse_matrix(self.args["<contact_map>"])
        map_out = self.args["<subsampled_map>"]
        subsampled = hcs.subsample_contacts(map_in, float(self.args["--prop"]))
        subsampled = subsampled.tocoo()
        hio.save_sparse_matrix(subsampled, map_out)


class Convert(AbstractCommand):
    """
    Convert between different Hi-C dataformats. Currently supports tsv (GRAAL),
    bedgraph2D (cooler) and DADE.

    usage:
        convert [--frags=FILE] [--binning=BIN] [--chroms=FILE]
                [--out=DIR] [--prefix=NAME] --from=FORMAT --to=FORMAT <contact_map>

    arguments:
        <contact_map> : The file containing the contact frequencies.

    options:
        -f, --frags=FILE          File containing the fragments coordinates. If
                                  not already in the contact map file.
        -b, --binning=INT[k|M|G]b Fixed bin size to use. Help reconstructing
                                  GRAAL files from a bedgraph2D (cooler) file.
        -c, --chroms=FILE         File containing the chromosome informations, if not
                                  already in the contact map file.
        -o, --out=DIR             The directory where output files must be written.
        -P, --prefix=NAME         A prefix by which the output filenames should start.
        -F, --from=FORMAT         The format from which to convert. [default: GRAAL]
        -T, --to=FORMAT           The format to which files should be converted. [default: cooler]
    """

    def GRAAL_cooler(self):
        mat = hio.load_sparse_matrix(self.mat_path)
        frags = pd.read_csv(self.frags_path, delimiter="\t")
        hio.save_bedgraph2d(mat, frags, self.out_mat)

    def GRAAL_DADE(self):
        mat = hio.load_sparse_matrix(self.mat_path)
        frags = pd.read_csv(self.frags_path, delimiter="\t")
        annot = frags.apply(lambda x: str(x.chrom) + "~" + str(x.start_pos), axis=1)
        hio.to_dade_matrix(mat, annotations=annot, filename=self.out_mat)

    def DADE_GRAAL(self):
        hio.dade_to_GRAAL(
            self.mat_path,
            output_matrix=self.out_mat,
            output_contigs=self.out_chr,
            output_frags=self.out_frags,
        )

    def cooler_GRAAL(self):
        mat, frags = hio.load_bedgraph2d(self.mat_path, bin_size=self.binning)
        hio.save_sparse_matrix(mat, self.out_mat)
        frags.to_csv(self.out_frags, sep="\t", index=False)

    def execute(self):
        in_fmt = self.args["--from"]
        fun_map = {
            "GRAAL-cooler": self.GRAAL_cooler,
            "GRAAL-DADE": self.GRAAL_DADE,
            "DADE-GRAAL": self.DADE_GRAAL,
            "cooler-GRAAL": self.cooler_GRAAL,
        }
        self.mat_path = self.args["<contact_map>"]
        self.frags_path = self.args["--frags"]
        self.chroms_path = self.args["--chroms"]
        out_fmt = self.args["--to"]
        out_path = self.args["--out"]
        os.makedirs(out_path, exist_ok=True)
        prefix = self.args["--prefix"]
        if self.args["--binning"] is not None:
            self.binning = parse_bin_str(self.args["--binning"])
        else:
            self.binning = self.args["--binning"]

        try:
            conv = "-".join([in_fmt, out_fmt])
            conv_fun = fun_map[conv]
        except KeyError:
            logger.error("Conversion not implemented or unknown format")
            sys.exit(1)

        if out_fmt == "GRAAL":
            mat_name = (
                prefix + ".mat.tsv" if prefix else "abs_fragments_contacts_weighted.txt"
            )
            frags_name = prefix + ".frag.tsv" if prefix else "fragments_list.txt"
            chr_name = prefix + ".chr.tsv" if prefix else "info_contigs.txt"
            self.out_mat = join(out_path, mat_name)
            self.out_frags = join(out_path, frags_name)
            self.out_chr = join(out_path, chr_name)
        elif out_fmt == "cooler":
            mat_name = prefix + ".mat.bg2" if prefix else "cooler.mat.bg2"
            self.out_mat = join(out_path, mat_name)
        elif out_fmt == "DADE":
            mat_name = prefix + ".DADE.tsv" if prefix else "DADE.mat.tsv"
            self.out_mat = join(out_path, mat_name)

        conv_fun()


class Distancelaw(AbstractCommand):
    """Distance law tools.
    Take the distance law file from hicstuff and can average it, normalize it compute the
    slope of the curve and plot it.
    
    usage:
        distancelaw [--average] [--big-arm-only] [--inf=INT] [--sup=INT] [--outputfile=IMG] 
                    [--labels=DIR] --dist-tbl=FILE1[,FILE2,...]
    
    options:
        -a, --average                       If set, calculate the average of the distance 
                                            law of the different chromosomes/arms in each
                                            condition. If two file given average is
                                            mandatory.
        -b, --big-arm-only                  If set will take only the arms bigger than sup.
                                            You have to give a value for sup is set.
        -d, --dist-tbl=FILE1[,FILE2,...]    Directory to the file or files containing the 
                                            compute distance law. File should have the same
                                            format than the ones made by hicstuff pipeline.
        -i, --inf=INT                       Inferior born to plot the distance law. By 
                                            default the value is 3000 bp (3 kb). Have to
                                            be strictly positive.
        -l, --labels=STR1,STR2...           List of string of the labels for the plot 
                                            separated by a coma. If no labels given, give 
                                            the names "Sample 1", "Sample 2"...
        -o, --outputfile=IMG                Output file. Format must be compatible with 
                                            plt.savefig. Default : ./disatnce_law_plot.svg.
        -s, --sup=INT                       Superior born to plot the distance law. By 
                                            default the value is the maximum length of all
                                            the dataset given. Also if big arm only set, it
                                            will be the minimum size of the arms/chromosomes
                                            taken to make the average.
    """

    def execute(self):
        # Give the current directory as out_dir if no out_dir is given.
        if self.args["--outputfile"]:
            output_file = self.args["--outputfile"]
        else:
            output_file = None
        # Add the option big army only.
        if self.args["--big-arm-only"]:
            big_arm_only = True
        else:
            big_arm_only = False
        # Put the inf and sup according to the arguments given.
        if self.args["--inf"]:
            inf = int(self.args["--inf"])
        else:
            inf = 3000
        if self.args["--sup"]:
            sup = int(self.args["--sup"])
        # Put in a list the path or the different paths given.
        distance_law_file = self.args["--dist-tbl"]
        distance_law_files = distance_law_file.split(",")
        length_files = len(distance_law_files)
        # Make new lists for the modified distance law.
        xs = [None] * length_files
        ps = [None] * length_files
        names = [None] * length_files
        # Sanity check : Average mandatory if more than one file.
        if not self.args["--average"] and length_files > 1:
            logger.error("You have to average if more than one file.")
            sys.exit(1)
        # Iterate on the different file given by the user.
        for i in range(length_files):
            xs[i], ps[i], names[i] = hcdl.import_distance_law(distance_law_files[i])
            # Make the average if enabled
            if self.args["--average"]:
                xs[i], ps[i] = hcdl.average_distance_law(
                    xs[i], ps[i], sup, big_arm_only
                )
                # If not average, we should to remove one level of list to have the good dimension.
        if not self.args["--average"]:
            names = names[0]
            xs = xs[0]
            ps = ps[0]
        # Normalize and make the derivative
        ps = hcdl.normalize_distance_law(xs, ps)
        slope = hcdl.slope_distance_law(xs, ps)
        # Gave new names for the different samples.
        if self.args["--labels"]:
            labels = self.args["--labels"]
            labels = labels.split(",")
        else:
            if length_files == 1 and not self.args["--average"]:
                labels = []
                for i in range(len(names)):
                    labels.append(names[i][0])
            else:
                labels = []
                for i in range(length_files):
                    labels.append("Sample " + str(i))
        # Make the plot if enabled, if not average plot the different arms or
        # chromosomes with the initial names else plot the different conditions
        # with the names labels.
        if not self.args["--sup"]:
            sup = max(max(xs, key=len))
        hcdl.plot_ps_slope(xs, ps, slope, labels, output_file, inf, sup)


def parse_bin_str(bin_str):
    """Bin string parsing

    Take a basepair binning string as input and converts it into
    corresponding basepair values.

    Parameters
    ----------
    bin_str : str
        A basepair region (e.g. 150KB). Unit can be BP, KB, MB, GB.

    Example
    -------

        >>> parse_bin_str("150KB")
        150000
        >>> parse_bin_str("0.1mb")
        100000

    Returns
    -------
    binning : int
        The number of basepair corresponding to the binning string.
    """
    try:
        binning = int(bin_str)
    except ValueError:
        bin_str = bin_str.upper()
        binsuffix = {"B": 1, "K": 1000, "M": 1e6, "G": 1e9}
        unit_pos = re.search(r"[KMG]?B[P]?$", bin_str).start()
        bp_unit = bin_str[unit_pos:]
        # Extract unit and multiply accordingly for fixed bp binning
        binning = int(float(bin_str[:unit_pos]) * binsuffix[bp_unit[0]])

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
    coord : tuple
        A tuple containing the bin range containing in the requested region.
    """
    if ":" in ucsc_str:
        chrom, bp = ucsc_str.split(":")
        bp = bp.replace(",", "").upper()
        start, end = bp.split("-")
        start, end = parse_bin_str(start), parse_bin_str(end)
        # Make absolute bin index (independent of chrom)
        bins["id"] = bins.index
        chrombins = bins.loc[bins.iloc[:, 0] == chrom, :]
        start = max([start, 1])
        start = max(chrombins.id[chrombins.iloc[:, 1] // start == 0])
        end = max(chrombins.id[chrombins.iloc[:, 1] // end == 0])
    else:
        chrom = ucsc_str
        # Make absolute bin index (independent of chrom)
        bins["id"] = bins.index
        chrombins = bins.loc[bins.iloc[:, 0] == chrom, :]
        try:
            start = min(chrombins.id)
            end = max(chrombins.id)
        except ValueError:
            logger.error("Invalid chromosome")
            raise
    coord = (int(start), int(end))
    return coord
