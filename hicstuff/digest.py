#!/usr/bin/env python3
# coding: utf-8

"""Genome digestion

Functions used to write auxiliary instaGRAAL compatible
sparse matrices.
"""

from Bio import SeqIO, SeqUtils
from Bio.Restriction import RestrictionBatch, Analysis
import os
import sys
import collections
import copy
import matplotlib.pyplot as plt
import pandas as pd
from hicstuff.log import logger

DEFAULT_FRAGMENTS_LIST_FILE_NAME = "fragments_list.txt"
DEFAULT_INFO_CONTIGS_FILE_NAME = "info_contigs.txt"
DEFAULT_SPARSE_MATRIX_FILE_NAME = "abs_fragments_contacts_weighted.txt"
DEFAULT_KB_BINNING = 1
DEFAULT_THRESHOLD_SIZE = 0
# Most used enzyme for eukaryotes
DEFAULT_ENZYME = "DpnII"
# If using evenly-sized chunks instead of restriction
# enzymes, they shouldn't be too short
DEFAULT_MIN_CHUNK_SIZE = 50


def write_frag_info(
    fasta,
    enzyme,
    size=DEFAULT_THRESHOLD_SIZE,
    circular=False,
    output_contigs=DEFAULT_INFO_CONTIGS_FILE_NAME,
    output_frags=DEFAULT_FRAGMENTS_LIST_FILE_NAME,
    output_dir=None,
):
    """Digest and write fragment information

    Write the fragments_list.txt and info_contigs.txt that are necessary for
    instaGRAAL to run.

    Parameters
    ----------
    fasta : pathlib.Path or str
        The path to the reference genome
    enzyme : str, int or list of str
        If a string, must be the name of an enzyme (e.g. DpnII) and the genome
        will be cut at the enzyme's restriction sites. If a number, the genome
        will be cut uniformly into chunks with length equal to that number. A
        list of enzymes can also be specified if using multiple enzymes.
    size : float, optional
        Size below which shorter contigs are discarded. Default is 0, i.e. all
        contigs are retained.
    circular : bool, optional
        Whether the genome is circular. Default is False.
    output_contigs : str, optional
        The name of the file with contig info. Default is info_contigs.txt
    output_frags : str, optional
        The name of the file with fragment info. Default is fragments_list.txt
    output_dir : [type], optional
        The path to the output directory, which will be created if not already
        existing. Default is the current directory.
    """

    try:
        enz = [enzyme] if type(enzyme) is str else enzyme
        my_enzyme = RestrictionBatch(enz)
    except ValueError:
        my_enzyme = max(int(enzyme), DEFAULT_MIN_CHUNK_SIZE)

    records = SeqIO.parse(fasta, "fasta")

    try:
        info_contigs_path = os.path.join(output_dir, output_contigs)
        frag_list_path = os.path.join(output_dir, output_frags)
    except AttributeError:
        info_contigs_path = output_contigs
        frag_list_path = output_frags

    with open(info_contigs_path, "w") as info_contigs:

        info_contigs.write("contig\tlength\tn_frags\tcumul_length\n")

        with open(frag_list_path, "w") as fragments_list:

            fragments_list.write(
                "id\tchrom\tstart_pos" "\tend_pos\tsize\tgc_content\n"
            )

            total_frags = 0

            for record in records:
                my_seq = record.seq
                contig_name = record.id
                contig_length = len(my_seq)
                n = len(my_seq)
                if contig_length < int(size):
                    continue
                try:
                    # Find sites of all restriction enzymes given
                    ana = Analysis(my_enzyme, my_seq, linear=not circular)
                    sites = ana.full()
                    # Gets all sites into a single flat list with 0-based index
                    sites = [
                        site - 1 for enz in sites.values() for site in enz
                    ]
                    # Sort by position and allow first add start and end of seq
                    sites.sort()
                    sites.insert(0, 0)
                    sites.append(n)
                    my_frags = (
                        my_seq[sites[i] : sites[i + 1]]
                        for i in range(len(sites) - 1)
                    )

                except TypeError:
                    my_frags = (
                        my_seq[i : min(i + my_enzyme, n)]
                        for i in range(0, len(my_seq), my_enzyme)
                    )
                n_frags = 0

                current_id = 1
                start_pos = 0
                for frag in my_frags:
                    size = len(frag)
                    if size > 0:
                        end_pos = start_pos + size
                        gc_content = SeqUtils.GC(frag) / 100.0

                        current_fragment_line = "%s\t%s\t%s\t%s\t%s\t%s\n" % (
                            current_id,
                            contig_name,
                            start_pos,
                            end_pos,
                            size,
                            gc_content,
                        )

                        fragments_list.write(current_fragment_line)

                        try:
                            assert (current_id == 1 and start_pos == 0) or (
                                current_id > 1 and start_pos > 0
                            )
                        except AssertionError:
                            logger.error((current_id, start_pos))
                            raise
                        start_pos = end_pos
                        current_id += 1
                        n_frags += 1

                current_contig_line = "%s\t%s\t%s\t%s\n" % (
                    contig_name,
                    contig_length,
                    n_frags,
                    total_frags,
                )
                total_frags += n_frags
                info_contigs.write(current_contig_line)


def intersect_to_sparse_matrix(
    intersect_sorted,
    fragments_list=DEFAULT_SPARSE_MATRIX_FILE_NAME,
    output_file=DEFAULT_SPARSE_MATRIX_FILE_NAME,
    output_dir=None,
    bedgraph=False,
):
    """Generate a GRAAL-compatible sparse matrix from a sorted intersection
    BED file.
    """

    try:
        output_file_path = os.path.join(output_dir, output_file)
    except AttributeError:
        output_file_path = output_file

    logger.info("Building fragment position dictionary...")
    # Build dictionary of absolute positions and fragment ids
    ids_and_positions = dict()
    with open(fragments_list) as fraglist_handle:
        _ = next(fraglist_handle)
        my_id = 0
        for line in fraglist_handle:
            contig_name, position, end = line.rstrip("\n").split("\t")[1:4]
            ids_and_positions[(contig_name, position, end)] = my_id
            my_id += 1
    logger.info("Done.")

    logger.info("Counting contacts...")

    # Detect and count contacts between fragments
    contacts = collections.Counter()
    with open(intersect_sorted) as intersect_handle:
        is_forward = True
        for line in intersect_handle:
            if is_forward:
                read_forward = line.rstrip("\n").split("\t")
                is_forward = False
                continue
            else:
                (
                    _,
                    start_forward,
                    end_forward,
                    name_forward,
                    orientation_forward,
                    contig_forward,
                    start_fragment_forward,
                    end_fragment_forward,
                ) = read_forward

                read_reverse = line.rstrip("\n").split("\t")
                (
                    _,
                    start_reverse,
                    end_reverse,
                    name_reverse,
                    orientation_reverse,
                    contig_reverse,
                    start_fragment_reverse,
                    end_fragment_reverse,
                ) = read_reverse

                # Detect contacts in the form of matching readnames
                # (last two characters are stripped in case read
                # name ends with '/1' or '/2')
                short_name_forward = name_forward.split()[0]
                short_name_reverse = name_reverse.split()[0]
                if short_name_forward == short_name_reverse:
                    abs_position_for = (
                        contig_forward,
                        start_fragment_forward,
                        end_fragment_forward,
                    )
                    abs_position_rev = (
                        contig_reverse,
                        start_fragment_reverse,
                        end_fragment_reverse,
                    )
                    try:
                        id_frag_for = ids_and_positions[abs_position_for]
                        id_frag_rev = ids_and_positions[abs_position_rev]
                    except KeyError:
                        logger.warning(
                            (
                                "Couldn't find matching fragment "
                                "id for position {} or position "
                                "{}".format(abs_position_for, abs_position_rev)
                            ),
                            file=sys.stderr,
                        )
                    else:
                        fragment_pair = tuple(
                            sorted((id_frag_for, id_frag_rev))
                        )
                        contacts[fragment_pair] += 1
                        # print("Successfully added contact between"
                        #       " {} and {}".format(id_fragment_forward,
                        #                           id_fragment_reverse))
                    finally:
                        is_forward = True
                else:
                    # If for some reason some reads are not properly
                    # interleaved, just skip the previous line and
                    # move on with the current line
                    # print("Read name {} does not match successor {}, "
                    # "reads are not properly interleaved".format(name_forward,
                    #                                            name_reverse))
                    read_forward = copy.deepcopy(read_reverse)
                    is_forward = False
    logger.info("Done.")

    logger.info("Writing sparse matrix...")
    if bedgraph:
        # Get reverse mapping between fragments ids and pos
        positions_and_ids = {
            id: pos for pos, id in list(ids_and_positions.items())
        }

        def parse_coord(coord):
            return "\t".join(str(x) for x in coord)

        with open(output_file_path, "w") as output_handle:
            for id_pair in sorted(contacts):
                id_fragment_a, id_fragment_b = id_pair
                nb_contacts = contacts[id_pair]
                coord_a = parse_coord(positions_and_ids[id_fragment_a])
                coord_b = parse_coord(positions_and_ids[id_fragment_b])
                line_to_write = "{}\t{}\t{}\n".format(
                    coord_a, coord_b, nb_contacts
                )
                output_handle.write(line_to_write)

    else:
        with open(output_file_path, "w") as output_handle:
            output_handle.write("id_frag_a\tid_frag_b\tn_contact\n")
            for id_pair in sorted(contacts):
                id_fragment_a, id_fragment_b = id_pair
                nb_contacts = contacts[id_pair]
                line_to_write = "{}\t{}\t{}\n".format(
                    id_fragment_a, id_fragment_b, nb_contacts
                )
                output_handle.write(line_to_write)

    logger.info("Done.")


def frag_len(
    output_frags=DEFAULT_FRAGMENTS_LIST_FILE_NAME,
    output_dir=None,
    plot=False,
    fig_path=None,
):
    """
    Generates summary of fragment length distribution based on an
    input fragment file. Can optionally show a histogram instead
    of text summary.
    Parameters
    ----------
    output_frags : str
        Path to the output list of fragments.
    output_dir : str
        Directory where the list should be saved.
    plot : bool
        Wether a histogram of fragment length should be shown.
    fig_path : str
        If a path is given, the figure will be saved instead of shown.
    """

    try:
        frag_list_path = os.path.join(output_dir, output_frags)
    except AttributeError:
        frag_list_path = output_frags
    frags = pd.read_csv(frag_list_path, sep="\t")
    nfrags = frags.shape[0]
    med_len = frags["size"].median()
    nbins = 40
    if plot:
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(frags["size"], bins=nbins)

        ax.set_xlabel("Fragment length [bp]")
        ax.set_ylabel("Log10 number of fragments")
        ax.set_title("Distribution of restriction fragment length")
        ax.set_yscale("log", basey=10)
        ax.annotate(
            "Total fragments: {}".format(nfrags),
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax.annotate(
            "Median length: {}".format(med_len),
            xy=(0.95, 0.90),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="top",
        )
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        if fig_path:
            plt.savefig(fig_path)
        else:
            plt.show()
    else:
        logger.info(
            "Genome digested into {0} fragments with a median "
            "length of {1}".format(nfrags, med_len)
        )
