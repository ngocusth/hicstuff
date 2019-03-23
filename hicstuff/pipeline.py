"""
Handle generation of GRAAL-compatible contact maps from fastq files.
cmdoret, 20190322
"""
import os
import time
from os.path import join
import subprocess as sp
from Bio import SeqIO
import hicstuff.digest as hcd
import hicstuff.iteralign as hci
import hicstuff.filter as hcf


def align_reads(
    reads,
    genome,
    out_sam,
    tmp_dir=None,
    threads=1,
    minimap2=False,
    iterative=False,
):
    """
    Select and call correct alignment method and generate logs accordingly.

    Parameters
    ----------
    reads : str
        Path to the fastq file with Hi-C reads.
    genome : str
        Path to the genome in fasta format
    out_sam : str
        Path to the output SAM file containing mapped Hi-C reads.
    tmp_dir : str
        Path where temporary files are stored.
    threads : int
        Number of threads to run alignments in parallel.
    aligner : bool
        Use minimap2 instead of bowtie2.
    iterative : bool
        Wether to use the iterative mapping procedure (truncating reads and
        extending iteratively)
    """
    if tmp_dir is None:
        tmp_dir = os.getcwd()
    index = None

    if iterative:
        hci.temp_directory = hci.generate_temp_dir(tmp_dir)
        hci.iterative_align(
            reads, tmp_dir=tmp_dir, ref=genome, n_cpu=threads, sam_out=out_sam
        )
    else:
        if minimap2:
            map_cmd = "minimap2 -2 -t {threads} -ax sr {fasta} {reads} > {sam}"
        else:
            index = hci.check_bt2_index(genome)
            map_cmd = "bowtie2 --very-sensitive-local -p {threads} -x {index} -U {fastq} > {sam}"
        map_args = {
            "threads": threads,
            "sam": out_sam,
            "fastq": "reads",
            "fasta": genome,
            "index": index,
        }
        sp.call(map_cmd.format(map_args))


def sam2pairs(sam1, sam2, out_pairs, min_qual=30):
    """
    Make a .pairs file from two Hi-C sam files. The Hi-C mates are matched
    by read identifier. Pairs where at least one reads maps with MAPQ below 
    min_qual threshold are discarded.

    Parameters
    ----------
    sam1 : str
        Path to the SAM file with aligned Hi-C reads.
    sam2 : str
        Path to the SAM file with aligned Hi-C reads.
    out_pairs : str
        Path to the output space-separated .pairs file with columns 
        readID, chr1 pos1 chr2 pos2 strand1 strand2
    """
    # Write header lines
    ...


def generate_matrix(pairs, mat):
    """Generate the matrix by counting the number of occurences of each
    combination of restriction fragments in a 2D BED file.

    Parameters
    ----------
    pairs : str
        Path to a Hi-C pairs file in 2D BED format.
    mat : str
        Path where the matrix will be written.
    """
    ...


def full_pipeline(
    input1,
    genome,
    input2=None,
    enzyme=5000,
    circular=False,
    out_dir=None,
    tmp_dir=None,
    plot=False,
    min_qual=30,
    min_size=0,
    threads=1,
    no_cleanup=False,
    iterative=False,
    filter_events=False,
    prefix=None,
    start_stage="fastq",
    bedgraph=False,
    minimap2=False,
):
    """
    Run the whole hicstuff pipeline. Starting from fastq files and a genome to
    obtain a contact matrix.
    
    Parameters
    ----------
    input1 : str
        Path to the Hi-C reads in fastq format (forward), the aligned Hi-C reads
        in SAM format, or the pairs file, depending on the value of start_stage.
    reads2 : str
        Path to the Hi-C reads in fastq format (forward), the aligned Hi-C reads
        in SAM format, or None, depending on the value of start_stage.
    genome : str
        Path to the genome in fasta format.
    enzyme : int or str
        Name of the enzyme used for the digestion (e.g "DpnII"). If an integer
        is used instead, the fragment attribution will be done directly using a
        fixed chunk size.
    circular : bool
        Use if the genome is circular.
    out_dir : str or None
        Path where output files should be written. Current directory by default.
    tmp_dir : str or None
        Path where temporary files will be written. Creates a "tmp" folder in
        out_dir by default.
    plot : bool
        Whether plots should be generated at different steps of the pipeline.
        Plots are saved in a "plots" directory inside out_dir.
    min_qual : int
        Minimum mapping quality required to keep a pair of Hi-C reads.
    min_size : int
        Minimum fragment size required to keep a restriction fragment.
    threads : int
        Number of threads to use for parallel operations.
    no_cleanup : bool
        Whether temporary files should be deleted at the end of the pipeline.
    iterative : bool
        Use iterative mapping. Truncates and extends reads until unambiguous
        alignment.
    filter_events : bool
        Filter spurious or uninformative 3C events. Requires a restriction enzyme.
    prefix : str or None
        Choose a common name for output files instead of default GRAAL names.
    start_stage : str
        Step at which the pipeline should start. Can be "fastq", "sam" or "pairs".
    bedgraph : bool
        Use the cooler-compatible bedgraph2 format instead of GRAAL format when
        writing the matrix
    minimap2 : bool
        Use minimap2 instead of bowtie2 for read alignment.
    """
    # Pipeline can start from 3 input types
    stages = {"fastq": 0, "sam": 1, "pairs": 2}
    start_stage = stages[start_stage]

    if out_dir is None:
        out_dir = os.getcwd()

    if tmp_dir is None:
        tmp_dir = join(out_dir, "tmp")
    os.mkdir(out_dir)
    os.mkdir(tmp_dir)

    # Define figures output paths
    if plot:
        fig_dir = join(out_dir, "plots")
        frag_plot = join(fig_dir, "frags_hist.pdf")
        dist_plot = join(fig_dir, "event_distance.pdf")
        pie_plot = join(fig_dir, "event_distribution.pdf")
    else:
        fig_dir = None
        dist_plot = pie_plot = frag_plot = None

    # Use current time for logging and to identify files
    now = time.strftime("%Y%m%d%H%M%S")

    def _tmp_file(fname):
        if prefix:
            fname = prefix + "." + fname
        return join(tmp_dir, fname)

    def _out_file(fname):
        if prefix:
            fname = prefix + "." + fname
        return join(out_dir, fname)

    # Define temporary file names
    log_file = _out_file("hicstuff_" + now + ".log")
    sam1 = _tmp_file("for.sam")
    sam2 = _tmp_file("rev.sam")
    pairs = _tmp_file("pairs.tsv")
    pairs_idx = _tmp_file("pairs_idx.tsv")

    # Define output file names
    if prefix:
        fragments_list = _out_file("mat.tsv")
        info_contigs = _out_file("chr.tsv")
        mat = _out_file("mat.tsv")
    else:
        # Default GRAAL file names
        fragments_list = _out_file("fragments_list.txt")
        info_contigs = _out_file("info_contigs.txt")
        mat = _out_file("abs_fragments_contacts_weighted.txt")

    # Define what input files are given
    if start_stage == 0:
        reads1, reads2 = input1, input2
    elif start_stage == 1:
        sam1, sam2 = input1, input2
    elif start_stage == 2:
        pairs_idx = input1

    # Perform genome alignment
    if start_stage == 0:
        align_reads(
            reads1,
            genome,
            "end1.sam",
            tmp_dir=tmp_dir,
            threads=threads,
            minimap2=minimap2,
            iterative=iterative,
        )
        align_reads(
            reads2,
            genome,
            "end2.sam",
            tmp_dir=tmp_dir,
            threads=threads,
            minimap2=minimap2,
            iterative=iterative,
        )

        # Generate info_contigs and fragments_list output files
        hcd.write_frag_info(
            genome,
            enzyme,
            min_size=min_size,
            circular=circular,
            output_contigs=info_contigs,
            output_frags=fragments_list,
        )

        # Log fragment size distribution
        hcd.frag_len(
            frags_file_name=fragments_list, plot=plot, fig_path=frag_plot
        )

    if start_stage < 2:
        pairs = "pairs.bed"
        # Make pairs file (readID, chr1, chr2, pos1, pos2, strand1, strand2)
        sam2pairs(sam1, sam2, pairs)
        restrict_table = {}
        for record in SeqIO.parse(genome, "fasta"):
            restrict_table[record.id] = hcd.get_restriction_table(
                record.seq, enzyme, circular=circular
            )

        # Add fragment index to pairs (readID, chr1, pos1, chr2,
        # pos2, strand1, strand2, frag1, frag2)
        hcd.attribute_fragments(pairs, pairs_idx, restrict_table)

    if filter_events:
        uncut_thr, loop_thr = hcf.get_thresholds(pairs_idx)
        hcf.filter_events(
            pairs_idx,
            pairs_idx,
            uncut_thr,
            loop_thr,
            fig_path=pie_plot,
            prefix=prefix,
        )

    # NOTE: hicstuff.digest module has an "intersect_to_sparse_matrix". Could
    # start from this and make it work with pairs files.
    generate_matrix(pairs)
