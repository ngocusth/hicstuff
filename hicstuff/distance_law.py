#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib as plt
import warnings
from scipy import ndimage
from matplotlib import cm
import hicstuff.io as hio
import pandas as pd
import os as os
import csv as csv


def export_distance_law(xs, ps, names, out_dir=None):
    """ Export the xs and ps from two list of numpy.ndarrays to a table in txt 
    file with three coulumns separated by a tabulation. The first column
    contains the xs, the second the ps and the third the name of the arm or 
    chromosome. The file is createin the directory given by outdir or the 
    current directory if no directory given.
    
    Parameters
    ----------
    xs : list of numpy.ndarray
        The list of the logbins of each ps.
    ps : list of numpy.ndarray
        The list of ps.
    names : list of string
        List containing the names of the chromosomes/arms/conditions of the ps
        values given.
    out_dir : str or None
        Path where output files should be written. Current directory by 
        default.
    
    Return
    ------
    txt file:
         File with three coulumns separated by a tabulation. The first column
         contains the xs, the second the ps and the third the name of the arm  
         or chromosome. The file is createin the directory given by outdir or 
         the current directory if no directory given. 
    """
    # Give the current directory as out_dir if no out_dir is given.
    if out_dir is None:
        out_dir = os.getcwd()
    # Sanity check: as many chromosomes/arms as ps
    if len(xs) != len(names):
        sys.stderr.write("ERROR: Number of chromosomes/arms and number of ps differ.")
        sys.exit(1)
    # Create the file and write it
    f = open(out_dir, "w")
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            ligne = str(xs[i][j]) + "\t" + str(ps[i][j]) + "\t" + names[i] + "\n"
            f.write(ligne)
    f.close()


def import_distance_law(distance_law_file):
    """ Import the table create by export_distance_law and return the list of 
    xs and ps in the order of the chromosomes.
    
    Parameters
    ----------
    distance_law_file : string
        Path to the file containing three columns : the xs, the ps, and the 
        chromosome/arm name.
    
    Return
    ------
    list of numpy.ndarray :
        The start coordinate of each bin one array per chromosome or arm.
    list of numpy.ndarray :
        The distance law probabilities corresponding of the bins of the 
        previous list.
    """
    file = pd.read_csv(distance_law_file, sep="\t", header=0)
    names = list(set(file.iloc[:, 2]))
    xs = [None] * len(names)
    ps = [None] * len(names)
    for i in range(len(names)):
        subfile = file[file.iloc[:, 2] == names[i]]
        xs[i] = np.array(subfile.iloc[:, 0])
        ps[i] = np.array(subfile.iloc[:, 1])
    return xs, ps


def get_chr_segment_bins_index(fragments, centro_file=None):
    """Get the index positions of the bins of different chromosomes, or arms if
    the centromers position have been given from the fragments file made by 
    hicstuff.
    
    Parameters
    ----------
    fragments : pandas.DataFrame
        Table containing in the first coulum the ID of the fragment, in the 
        second the names of the chromosome in the third and fourth the start 
        position and the end position of the fragment. The file have no header.
        (File like the 'fragments_list.txt' from hicstuff)
    centro_file : None or str
        None or path to a file with the genomic positions of the centromers 
        sorted as the chromosomes separated by a space. The file have only one 
        line.
        
    Returns
    -------
    list of floats :
        The start indices of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
    """
    # Get bins where chromosomes start
    chr_segment_bins = np.where(fragments == 0)[0]
    if centro_file is not None:
        # Read the file of the centromers
        with open(centro_file, "r", newline="") as centro:
            centro = csv.reader(centro, delimiter=" ")
            centro_pos = next(centro)
        # Sanity check: as many chroms as centromeres
        if len(chr_segment_bins) != len(centro_pos):
            sys.stderr.write("ERROR: Number of chromosomes and centromeres differ.")
            sys.exit(1)
        # Get bins of centromeres
        centro_bins = np.zeros(len(centro_pos))
        for i in range(len(chr_segment_bins)):
            if (i + 1) < len(chr_segment_bins):
                subfrags = fragments[chr_segment_bins[i] : chr_segment_bins[i + 1]]
            else:
                subfrags = fragments[chr_segment_bins[i] :]
            # index of last fragment starting before centro in same chrom
            centro_bins[i] = chr_segment_bins[i] + max(
                np.where((subfrags["start_pos"][:] // int(centro_pos[i])) == 0)[0]
            )
        # Combine centro and chrom bins into a single array. Values are start
        # bins of arms
        chr_segment_bins = np.sort(np.concatenate((chr_segment_bins, centro_bins)))
    return chr_segment_bins


def get_chr_segment_length(fragments, chr_segment_bins):
    """Compute a list of the length of the different objects (arm or 
    chromosome) given by chr_segment_bins.
    
    Parameters
    ----------
    fragments : pandas.DataFrame
        Table containing in the first coulum the ID of the fragment, in the 
        second the names of the chromosome in the third and fourth the start 
        position and the end position of the fragment. The file have no header.
        (File like the 'fragments_list.txt' from hicstuff)
    chr_segment_bins : list of floats
        The start position of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
        
    Returns
    -------
    list of numpy.ndarray:
        The length in base pairs of each chromosome or arm.
    """
    chr_segment_length = [None] * len(chr_segment_bins)
    # Iterate in chr_segment_bins in order to obtain the size of each chromosome/arm
    for i in range(len(chr_segment_bins) - 1):
        # Obtain the size of the chromosome/arm, the if loop is to avoid the
        # case of arms where the end position of the last fragments doesn't
        # mean th size of arm. If it's the right we have to remove the size of
        # the left arm.
        if (
            fragments["start_pos"].iloc[int(chr_segment_bins[i])] == 0
            or fragments["start_pos"][int(chr_segment_bins[i])] == 1
        ):
            n = fragments["end_pos"].iloc[int(chr_segment_bins[i + 1]) - 1]
        else:
            n = (
                fragments["end_pos"].iloc[int(chr_segment_bins[i + 1]) - 1]
                - fragments["end_pos"].iloc[int(chr_segment_bins[i]) - 1]
            )
        chr_segment_length[i] = n
    # Case of the last xs where we take the last end position
    if (
        fragments["start_pos"][int(chr_segment_bins[-2])] == 0
        or fragments["start_pos"][int(chr_segment_bins[-2])] == 1
    ):
        n = fragments["end_pos"].iloc[-1]
    else:
        n = (
            fragments["end_pos"].iloc[-1]
            - fragments["end_pos"].iloc[int(chr_segment_bins[-2]) - 1]
        )
    chr_segment_length[-1] = n
    return chr_segment_length


def logbins_xs(
    fragments, chr_segment_bins, chr_segment_length, base=1.1, circular=False
):
    """Compute the logbins of each chromosome/arm in order to have theme to
    compute distance law. At the end you will have bins of increasing with a 
    logspace with the base of the value given in base.
    
    Parameters
    ----------
    fragments : pandas.DataFrame
        Table containing in the first coulum the ID of the fragment, in the 
        second the names of the chromosome in the third and fourth the start 
        position and the end position of the fragment. The file have no header.
        (File like the 'fragments_list.txt' from hicstuff)
    chr_segment_bins : list of floats
        The start position of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
    chr_segment_length: list of floats
        List of the size in base pairs of the different arms or chromosomes.
    base : float
        Base use to construct the logspace of the bins, 1.1 by default.
    circular : bool
        If True, calculate the distance as the chromosome is circular. Default 
        value is False.
        
    Returns
    -------
    list of numpy.ndarray :
        The start coordinate of each bin one array per chromosome or arm.
    """
    # Create the xs array and a list of the length of the chromosomes/arms
    xs = [None] * len(chr_segment_bins)
    # Iterate in chr_segment_bins in order to make the logspace
    for i in range(len(chr_segment_length)):
        n = chr_segment_length[i]
        # if the chromosome is circular the mawimum distance between two reads
        # are divided by two
        if circular:
            n /= 2
        n_bins = int(np.log(n) / np.log(base) + 1)
        # For each chromosome/arm compute a logspace to have the logbin
        # equivalent to the size of the arms and increasing size of bins
        xs[i] = np.unique(
            np.logspace(0, n_bins - 1, num=n_bins, base=base, dtype=np.int)
        )
    return xs


def circular_distance_law(distance, chr_segment_length, chr_bin):
    """Recalculate the distance to return the distance in a circular chromosome
    and not the distance between the two genomic positions.
    
    Parameters
    ----------
    chr_segment_bins : list of floats
        The start position of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
    chr_segment_length: list of floats
        List of the size in base pairs of the different arms or chromosomes.
    distance : int
        Distance between two fragments with a contact.
        
    Returns
    -------
    int :
        The real distance in the chromosome circular and not the distance 
        between two genomic positions
        
    Examples
    --------
    >>> circular_distance_law(7500, [2800, 9000], 1)
    1500
    >>> circular_distance_law(1300, [2800, 9000], 0)
    1300
    >>> circular_distance_law(1400, [2800, 9000], 0)
    1400
    """
    chr_len = chr_segment_length[chr_bin]
    if distance > chr_len / 2:
        distance = chr_len - distance
    return distance


def get_pairs_distance(
    line, fragments, chr_segment_bins, chr_segment_length, xs, ps, circular=False
):
    """From a line of a pair reads file, filter -/+ or +/- reads, keep only the 
    reads in the same chromosome/arm and compute the distance of the the two
    fragments. It modify the input ps in order to count or not the line given. 
    It will add one in the logbin corresponding to the distance.
    
    Parameters
    ----------
    line : OrderedDict 
        Line of a pair reads file with the these keys readID, chr1, pos1, chr2,
        pos2, strand1, strand2, frag1, frag2. The values are in a dictionnary.
    fragments : pandas.DataFrame
        Table containing in the first coulum the ID of the fragment, in the 
        second the names of the chromosome in the third and fourth the start 
        position and the end position of the fragment. The file have no header.
        (File like the 'fragments_list.txt' from hicstuff)
    chr_segment_bins : list of floats
        The start position of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
    chr_segment_length: list of floats
        List of the size in base pairs of the different arms or chromosomes.
    xs : list of lists
        The start coordinate of each bin one array per chromosome or arm.
    ps : list of lists
        The sum of contact already count. xs and ps should have the same 
        dimensions.
    circular : bool
        If True, calculate the distance as the chromosome is circular. Default 
        value is False.
    """
    # We only keep the event +/+ or -/-. This is done to avoid to have any
    # event of uncut which are not possible in these events. We can remove the
    # good events of +/- or -/+ because we don't need a lot of reads to compute
    # the distance law and if we eliminate these reads we do not create others
    # biases as they should have the same distribution.
    if line["strand1"] == line["strand2"]:
        # Find in which chromosome/arm are the fragment 1 and 2.
        chr_bin1 = (
            np.searchsorted(chr_segment_bins, int(line["frag1"]), side="right") - 1
        )
        chr_bin2 = (
            np.searchsorted(chr_segment_bins, int(line["frag2"]), side="right") - 1
        )
        # We only keep the reads with the two fragments in the same chromosome
        # or arm.
        if chr_bin1 == chr_bin2:
            # For the reads -/-, the fragments should be religated with both
            # their start position (position in the left on the genomic
            # sequence, 5'). For the reads +/+ it's the contrary. We compute
            # the distance as the distance between the two extremities which
            # are religated.
            if line["strand1"] == "-":
                distance = abs(
                    np.array(fragments["start_pos"][int(line["frag1"])])
                    - np.array(fragments["start_pos"][int(line["frag2"])])
                )
            if line["strand1"] == "+":
                distance = abs(
                    np.array(fragments["end_pos"][int(line["frag1"])])
                    - np.array(fragments["end_pos"][int(line["frag2"])])
                )
            if circular:
                distance = circular_distance_law(distance, chr_segment_length, chr_bin1)
            xs_temp = xs[chr_bin1][:]
            # Find the logbins in which the distance is and add one to the sum
            # of contact.
            ps_indice = np.searchsorted(xs_temp, distance, side="right") - 1
            ps[chr_bin1][ps_indice] += 1


def get_names(fragments, chr_segment_bins):
    """Make a list of the names of the arms or the chromosomes.
    
    Parameters
    ----------
    fragments : pandas.DataFrame
        Table containing in the first coulum the ID of the fragment, in the 
        second the names of the chromosome in the third and fourth the start 
        position and the end position of the fragment. The file have no header.
        (File like the 'fragments_list.txt' from hicstuff)
    chr_segment_bins : list of numpy.ndarray
        The start position of chromosomes/arms to compute the distance law on 
        each chromosome/arm separately.
        
    Returns
    -------
    list of floats : 
        List of the labels given to the curves. It will be the name of the arms
        or chromosomes.
    """
    # Get the name of the chromosomes.
    chr_names = np.unique(fragments["chrom"])
    # Case where they are separate in chromosomes
    if len(chr_segment_bins) == len(chr_names):
        names = chr_names
    # Case where they are separate in arms
    else:
        names = []
        for chr in chr_names:
            names.append(chr + "_left")
            names.append(chr + "_rigth")
    return names


def get_distance_law(
    pairs_reads_file,
    fragments_file,
    centro_file=None,
    base=1.1,
    outdir=None,
    circular=False,
):
    """Compute distance law as a function of the genomic coordinate aka P(s).
    Bin length increases exponentially with distance. Works on pairs file 
    format from 4D Nucleome Omics Data Standards Working Group. If the genome 
    is composed of several chromosomes and you want to compute the arms 
    separately, provide a file with the positions of centromers. Create a file 
    with three coulumns separated by a tabulation. The first column contains 
    the xs, the second the ps and the third the name of the arm or chromosome. 
    The file is create in the directory given in outdir or in the current 
    directory if no directory given.
    
    Parameters
    ----------
    pairs_reads_file : string
        Path of a pairs file format from 4D Nucleome Omics Data Standards 
        Working Group with the 8th and 9th coulumns are the ID of the fragments
        of the reads 1 and 2.
    fragments_file : path
        Path of a table containing in the first column the ID of the fragment,
        in the second the names of the chromosome in the third and fourth 
        the start position and the end position of the fragment. The file have 
        no header. (File like the 'fragments_list.txt' from hicstuff)
    centro_file : None or str
        None or path to a file with the genomic positions of the centromers 
        sorted as the chromosomes separated by a space. The file have only one 
        line.
    base : float
        Base use to construct the logspace of the bins - 1.1 by default.
    outdir : None or str
        Directory of the output file. If no directory given, will be replace by
        the current directory.
    circular : bool
        If True, calculate the distance as the chromosome is circular. Default 
        value is False. Cannot be True if centro_file is not None     
    """
    # Sanity check : centro_fileition should be None if chromosomes are
    # circulars (no centromeres is circular chromosomes).
    if circular and centro_file != None:
        print("Chromosomes cannot have a centromere and be circular")
        raise ValueError
    # Import third columns of fragments file
    fragments = pd.read_csv(fragments_file, sep="\t", header=0, usecols=[0, 1, 2, 3])
    # Calculate the indice of the bins to separate into chromosomes/arms
    chr_segment_bins = get_chr_segment_bins_index(fragments, centro_file)
    # Calculate the length of each chromosoms/arms
    chr_segment_length = get_chr_segment_length(fragments, chr_segment_bins)
    xs = logbins_xs(fragments, chr_segment_bins, chr_segment_length, base, circular)
    # Create the list of p(s) with one array for each chromosome/arm and each
    # array contain as many values as in the logbin
    ps = [None] * len(chr_segment_bins)
    for i in range(len(xs)):
        ps[i] = [0] * len(xs[i])
    # Read the pair reads file
    with open(pairs_reads_file, "r", newline="") as reads:
        # Remove the line of the header
        header_length = len(hio.get_pairs_header(pairs_reads_file))
        for i in range(header_length):
            next(reads)
        # Reads all the others lines and put the values in a dictionnary with
        # the keys : 'readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1',
        # 'strand2', 'frag1', 'frag2'
        reader = csv.DictReader(
            reads,
            fieldnames=[
                "readID",
                "chr1",
                "pos1",
                "chr2",
                "pos2",
                "strand1",
                "strand2",
                "frag1",
                "frag2",
            ],
            delimiter=" ",
        )
        for line in reader:
            # Iterate in each line of the file after the header
            get_pairs_distance(
                line, fragments, chr_segment_bins, chr_segment_length, xs, ps, circular
            )
    # Divide the number of contacts by the area of the logbin
    for i in range(len(xs)):
        n = chr_segment_length[i]
        for j in range(len(xs[i]) - 1):
            # Use the area of a trapezium to know the area of the logbin with n
            # the size of the matrix.
            ps[i][j] /= ((2 * n - xs[i][j + 1] - xs[i][j]) / 2) * (
                (1 / np.sqrt(2)) * (xs[i][j + 1] - xs[i][j])
            )
        # Case of the last logbin which is an isosceles rectangle triangle
        ps[i][-1] /= ((n - xs[i][-1]) ** 2) / 2
    names = get_names(fragments, chr_segment_bins)
    export_distance_law(xs, ps, names, outdir)
