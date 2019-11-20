# -*- coding: utf-8 -*-
"""Created on Tue Mar 27 19:03:08 2018.

@author: inspired by axel KournaK. Modifications Remi
Agglomerates the signal around positions of interest in HiC contact maps
Input: a path to a folder containing contact maps for indivual chromosomes.
            they must be named chr1.txt, chr2.txt etc...
       a list of positions of interest as a txt file with 3 columns.
            1st column = chromosome name, 2nd column = x coordinate, 3d column = y coordinate
Output: an agglomerated plot
"""
import argparse, os, sys, glob, pathlib
import numpy as np
import numpy.ma as ma
import random as rando
import pandas as pd
from copy import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


import hicstuff.view as hcv
import hicstuff.io as hio

# import hicstuff.hicstuff as hcs
# import hicstuff.digest as hcd
# import hicstuff.iteralign as hci
# import hicstuff.filter as hcf
# from hicstuff.version import __version__

# =======================================================================
# ARGUMENTS
# =======================================================================
# def parse_args():
#     """ Gets the arguments from the command line."""
#     parser = argparse.ArgumentParser()

#     # Positional arguments
#     parser.add_argument(
#         "folder", help="the folder containing individual chromosomes contact maps"
#     )
#     parser.add_argument(
#         "borders",
#         help='the file containing loops or borders positions. \
#                         Format must be "chr pos1 pos2". A comma-separated list of files can be given.',
#     )
#     parser.add_argument(
#         "-b", "--bins", type=int, required=True, help="the resolution (bin) in bp"
#     )

#     # Optional arguments
#     parser.add_argument(
#         "-cen",
#         "--centromers",
#         default=None,
#         help="file with infos about centromers. If not provided, you need to precise --no_arm",
#     )
#     parser.add_argument(
#         "-cs",
#         "--color_scale",
#         type=float,
#         default=0.2,
#         help="The colorscale of the results. Default: 0.2.",
#     )
#     parser.add_argument(
#         "-l",
#         "--list_chr",
#         default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16",
#         help="the comma-separated list of chromosomes to take into account. Default = chr1 to chr16 (names with 1 digit).",
#     )
#     parser.add_argument(
#         "--masks",
#         default=None,
#         help="The bottom-left part of the heatmaps can be masked. In this case, enter comma-separated floats corresponding to the ratio of the heatmap to mask. \
#                         The list of floats can be shorter than the number of borders files (the last figures will be unmasked).",
#     )
#     parser.add_argument(
#         "--metric",
#         choices=["mean", "median"],
#         default="mean",
#         help="use mean or median. Default = mean. Currently only mean is implemented.",
#     )
#     parser.add_argument(
#         "--mode",
#         choices=["loop", "loops", "border", "borders"],
#         default="loop",
#         help="Precise, in output file name, if loops or borders were analyzed. Default: loop.",
#     )
#     parser.add_argument(
#         "-n",
#         "--nb_iter",
#         type=int,
#         default=10,
#         help="The number iteration for the null models. Default = 10.",
#     )
#     parser.add_argument(
#         "--no_arms",
#         action="store_true",
#         help="Don't divide matrices into right and left arm.",
#     )
#     parser.add_argument(
#         "--no_disp",
#         action="store_true",
#         help="Don't display the results and only saves them.",
#     )
#     parser.add_argument(
#         "-o",
#         "--outimg",
#         default="agglomerated_signal",
#         help="name of the output image without extension",
#     )
#     parser.add_argument(
#         "--pileup",
#         action="store_true",
#         help="Only averages the contacts and don't compute the log-ratio with expected signal. Traditionally use cmap afmhot_r",
#     )
#     parser.add_argument(
#         "-s",
#         "--scale",
#         choices=["ratio", "logratio"],
#         default="logratio",
#         help="Take the ratio or the log-ratio. Works only if pileup is False. Default = logratio.",
#     )
#     parser.add_argument(
#         "--suptitle",
#         default=None,
#         help="suptitle to discribe a picture composed of several windows. Default = None",
#     )
#     parser.add_argument(
#         "-t",
#         "--titles",
#         default=None,
#         help="Comma-separated titles for the different border files. Must be as many as borders files. Default=None.",
#     )
#     parser.add_argument(
#         "--vmin",
#         type=float,
#         default=0,
#         help="Value min for the heatmap colorscale when doing pileups. Must be between 0 and 1. Default: 0",
#     )
#     parser.add_argument(
#         "--vmax",
#         type=float,
#         default=0.8,
#         help="Value min for the heatmap colorscale when doing pileups. Must be between 0 and 1. Default: 0.8",
#     )
#     parser.add_argument(
#         "-w",
#         "--window",
#         default=40,
#         type=int,
#         help="the number of bins around the position to take into account. Default = 40.",
#     )
#     return parser.parse_args()


# =======================================================================
# FUNCTIONS
# =======================================================================
# =============================================
# Algoritmh backbone
# =============================================
def compute_pileups(
    matrix,
    contig_infos,
    positions,
    ranges,
    white_list,
    black_list,
    binning,
    centromers,
    sep_arms,
    window_size,
    null_model,
    nb_iter,
    logratio,
    cs,
    vmin,
    vmax,
    masks,
    title,
    outimg,
):
    """Compute the pileup of positions of interests in a given contact map.

    Parameters
    ----------
    matrix : lil matrix
        a contact matrix in lil format
    contigs_infos : DataFrame
        a pandas DataFrame with the information about matrix chromosomes:
        chrm length n_frags cumul_length.
    positions : DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance.
    ranges : list or None
        a list of tuples corresponding to ranges for stratification of positions of
        interest and run several pileup in one go.
    white_list : list or None
        a list of string corresponding to the chromosomes to include in the analysis.
        Their names must be the same as in the positions bg2 file. All other
        chromosomes will be excluded. Argument white_list has priority over argument
        black_list.
    black_list : list or None
        a list of string corresponding to the chromosomes to exclude from the analysis.
        Their names must be the same as in the positions bg2 file. All other
        chromosomes will be included.
    binning: int
        the binning of the contact maps.
    centromers : pandas DataFrame
        dataframe containing the position of the centromers of each chromosome.
    sep_arms: bool
        a boolean indicating if the positions in the inter-arm space must be excluded.
        If the analysed positions are on the main diagonal, sep_arms must be False. Default: True.
    window_size: int
        the size of the window to slice around the positions of interest (in pixels).Default: 40.
    null_model: bool
        a boolean indicating if the signal must be divided by the null model. Default: True.
    nb_iter: int
        the number of iteration when computing the null model
    logratio: boolean
        indicates if the result will be the ratio or the logratio of signal/null_model
    cs: float
        the colorscale for the ratios or logratios. The colorscale will be (-cs, +cs) if the
        argument logratio is True and (1-cs, 1+cs) if the argument logratio is False.
        The argument null_model must be True to take cs into account.
        If it is False, the colorscale can be tune with the arguments vmin and vmax.
    vmin: float
        the vmin value for the colorscale.
        The argument null_model must be False to take vmin into account.
        If it is True, the colorscale can be tuned with the argument cs. Default: 0.
    vmax: float
        the vmax value for the colorscale.
        The argument null_model must be False to take vmax into account.
        If it is False, the colorscale can be tuned with the argument cs. Default: 0.8.
    masks: list or None
        a list of floats indicating the part of each ax to mask.
        The masking starts from the lower-left corner in direction of the upper-left corner.
        If there are less masks values than ranges, the last axes will not be masked.
        Default: None.
    title: str or None
        the title displayed at to top center of the figure.
    outimg: str
        The name of the final figure. Default: "pileup"

    Returns
    -------
    result_signal: numpy array
        the pileup computed (signal or signal/null_model or log(signal/null_model))
    positions_count: int
        the number of positions agglomerated to compute the result_signal
    my_fig: matplotlib.pyplot figure
        the representation of result_signal
    """
    ## where to store the pileups
    list_pileups = []
    list_nb_positions = []

    # Prepare the list of positions: remove inter-arms if needed, keep only wanted chromosomes
    if sep_arms:
        positions = remove_inter_arms(positions, centromers)
    if isinstance(white_list, list):
        positions = keep_white_list(positions, white_list)
    elif isinstance(black_list, list):
        positions = remove_black_list(positions, black_list)

    # From positionns in bp in the bedgraph2 file deduce positions in bins
    positions["pos1"] = positions["start1"] + positions["end1"] // (2 * binning)
    positions["pos2"] = positions["start2"] + positions["end2"] // (2 * binning)

    # If no ranges are specified, create ranges as a list of size 1
    # containing a range large enough to include all positions
    if not isinstance(ranges, list):
        ranges = [(min(positions["distance"]), max(positions["distance"]))]

    # Get the limits of the different matrices to analyse
    matrices_limits = get_matrices_limits(
        matrix, contigs_infos, centromers, positions, window_size, sep_arms
    )

    # Compute the pileups (1 by range in the list ranges)
    for i, tup in enumerate(ranges):
        ## compute the signal matrix
        mat_agglo1, nb_positions = get_agglomerated_signal(
            matrix, positions, binning, window_size, sep_arms
        )

        ## compute the background matrix if needed
        if null_model:
            mat_agglo2 = get_null_model(matrix, positions, df, window, nb_iter, no_arms)
            mat_total = np.divide(mat_agglo1, mat_agglo2)
            if logratio:
                mat_total = np.log2(mat_total)
        else:
            mat_total = mat_agglo1

        list_pileups.append(mat_total)
        list_nb_positions.append(nb_positions)

    # Compute loops strength in each pileup
    loops_strengths = compute_score(list_pileups, logratio)

    # Plot the pileups
    my_fig = plot_agglo(
        list_pileups,
        list_nb_positions,
        loops_strengths,
        ranges,
        binning,
        window_size,
        null_model,
        cs,
        vmin,
        vmax,
        title,
        masks,
    )

    return result_signal, positions_count, my_fig


# =============================================
# Getting the data
# =============================================
# def load_matrices(bank_name, list_chr, centromers, no_arms):
#     """Take a list of matrices as .txt file and return them as a list of numpy
#     arrays.

#     Each matrix is divided into left and right arms. The list of correct indices is also returned.
#     Input: a list of contact maps as plain txt files
#     Output: a dictionary with matrices: dico = {chrm: {'matL': x, 'matR': x, 'indicesL': x, 'indicesR': x, 'centro': x}}
#     """
#     matrices = {}
#     for chrm in list_chr:
#         matrices[chrm] = {}
#         mat, indices = get_matrice_and_correct_indices(bank_name, chrm)
#         if no_arms:
#             matrices[chrm]["mat"] = mat
#             matrices[chrm]["indices"] = indices
#         else:
#             cen = int(centromers[centromers.chrm == chrm]["centro"])
#             matL, matR = divide_matrix(mat, cen)
#             indicesL, indicesR = adjust_indices(indices, cen)
#             matrices[chrm]["matL"] = matL
#             matrices[chrm]["matR"] = matR
#             matrices[chrm]["indicesL"] = indicesL
#             matrices[chrm]["indicesR"] = indicesR
#             matrices[chrm]["centro"] = cen
#     return matrices


# def get_matrice_and_correct_indices(bank_name, chrm):
#     """Returns the matrices present in the folder "bank_name" and a list of
#     correct indices, here indices above a given threshold."""
#     ## load raw and normalized matrices
#     mat_name = bank_name + chrm + "_raw.txt"
#     matraw = np.loadtxt(mat_name, dtype=float)
#     mat_name = bank_name + chrm + "_norm.txt"
#     matscn = np.loadtxt(mat_name, dtype=float)

#     ##compute the indices that can be taken, i.e. above a threshold
#     thres_low = np.median(matraw.sum(axis=0)) - np.std(matraw.sum(axis=0))
#     thres_sup = np.median(matraw.sum(axis=0)) + np.std(matraw.sum(axis=0))
#     # correct_indices = set(range(matscn.shape[0]))
#     correct_indices = set(np.where((matraw.sum(axis=0) > thres_low))[0])
#     # correct_indices = set(np.where((matraw.sum(axis=0) > thres_low) & (matraw.sum(axis=1) < thres_sup))[0])
#     return matscn, correct_indices


# def divide_matrix(mat, cen):
#     """
#     Divides a matrix into left and right arm
#     Input: a matrix as a numpy array
#            the position of the centromer
#     Output: 2 matrices as numpy arrays
#     """
#     matL = mat[: cen + 1, : cen + 1]
#     matR = mat[cen:, cen:]
#     return matL, matR


# def adjust_indices(indices, cen):
#     """Divides a list of indices int left and right arm.

#     Corrects their value
#     Input: a list of
#            the position of the centromer
#     Output: 2 matrices as numpy arrays
#     """
#     indicesL = [elt for elt in indices if elt < cen]
#     indicesR = [elt - cen for elt in indices if elt >= cen]
#     return indicesL, indicesR

# =============================================
# Filtering the positions
# =============================================
def remove_inter_arms(positions, centromers):
    """Remove from the DataFrame positions all the borders which
    are not on the same chromosome arm.

    Parameters
    ----------
    positions : pandas DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance
    centromers: pandas DataFrame
        a pandas DataFrame listing the positions of centromers for each chromosome
        chr1 centro

    Returns
    -------
    positions : pandas DataFrame
        a pandas DataFrame containing with only the intra-arm positions to study 
        in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance
    """
    positions = positions.merge(centromers, on=["chr1"], how="inner")
    positions = positions[
        (
            (
                (positions["end1"] <= positions["centro"])
                & (positions["end2"] <= positions["centro"])
            )
            | (
                (positions["start1"] > positions["centro"])
                & (positions["start2"] > positions["centro"])
            )
        )
    ]
    return positions.drop(columns=["centro"], inplace=True)


def keep_white_list(positions, white_list):
    """Remove from the DataFrame positions all the borders which
    are not in the list white_list

    Parameters
    ----------
    positions : pandas DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance.
    white_list : list
        a list of strings corresponding to the chromosomes to include in the analysis.
        Their names must be the same as in the positions DataFrame.
        All other chromosomes will be excluded.

    Returns
    -------
    positions : pandas DataFrame
        a pandas DataFrame containing with only the chromosomes to study
        in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance
    """
    return positions[
        (positions["chr1"].isin(white_list)) & (positions["chr2"].isin(white_list))
    ]


def remove_black_list(positions, black_list):
    """Remove from the DataFrame positions all the borders which
    are in the list black_list

    Parameters
    ----------
    positions : pandas DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance.
    black_list : list
        a list of strings corresponding to the chromosomes to include in the analysis.
        Their names must be the same as in the positions DataFrame.
        All other chromosomes will be excluded.

    Returns
    -------
    positions : pandas DataFrame
        a pandas DataFrame containing with only the chromosomes to study
        in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance
    """
    return positions[
        ~(positions["chr1"].isin(black_list)) & ~(positions["chr2"].isin(black_list))
    ]


# =============================================
# Computing the signal around positions of interest
# =============================================
def get_matrices_limits(
    matrix, contigs_infos, centromers, positions, window_size, sep_arms
):
    """Gets The limits of all matrices to analyse, generally either a whole chromosome 
    or the chromosomes arms. Return the results as a dictionary {chrm: [(pos, pos)]}

    Parameters
    ----------
    contigs_infos : DataFrame
        a pandas DataFrame with the information about matrix chromosomes:
        chrm length n_frags cumul_length
    centromers : pandas DataFrame
        dataframe containing the position of the centromers of each chromosome:
        chrm centro.
    positions : pandas DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance. Th condition chr1 == chr2 must
        be satisfied.
    sep_arms: bool
        a boolean indicating if the positions in the inter-arm space must be excluded.
        If the analysed positions are on the main diagonal, sep_arms must be False. Default: True.

    Returns
    -------
    matrices_limits: dict
        a dictionary containing all the chromosomes to analyse with their start and end:
        {chrm: [(pos, pos), (pos, pos)]}
    """
    # Deduce the end of the chromosomes from cumul_length and length
    contigs_infos["end"] = contigs_infos["cumul_length"] + contigs_infos["length"]

    # If sep arms, merge the dataframes contigs_infos and centromers on chromosome
    if sep_arms:
        contigs_infos.merge(centromers, on="chrm", how="inner")

    # Generate a dict containing, for each chromosome in positions, a list with
    # either the limits of the chromosome (start, end) or the limits of both arms
    # (start, centro) and (centro, end) if sep_arms is True.
    matrices_limits = {}
    for chrm in positions["chr1"].unique():
        matrices_limits[chrm] = []
        row = contigs_infos[contigs_infos["chrm"] == chrm].reset_index()
        if len(row) == 1:
            if sep_arms:
                matrices_limits[chrm].append(
                    (row.at[0, "cumul_length"], row.at[0, "centro"])
                )
                matrices_limits[chrm].append((row.at[0, "centro"], row.at[0, "end"]))
            else:
                matrices_limits[chrm].append(
                    (row.at[0, "cumul_length"], row.at[0, "end"])
                )

    return matrices_limits


def get_agglomerated_signal(matrix, matrices_limits, binning, positions, window_size):
    """Gets a sparse matrix and positions of interest and returns a matrix corresponding to
    the agglomeration of the signal around these positions.

    Parameters
    ----------
    matrix: lil matrix
        a contact matrix in lil format
    matrices_limits: dict
        a dictionary containing all the chromosomes to analyse with their start and end:
        {chrm: [(pos, pos), (pos, pos)]}
    binning: int
        a
    positions : pandas DataFrame
        a pandas DataFrame with all the positions to study in bedgraph2 format:
        chr1 start1 end1 chr2 start2 end2 distance. Th condition chr1 == chr2 must
        be satisfied.
    window_size : int
        the number of bins around the position to include in the analysis.
        The shape of the final signal matrix will be (2*window_size)x(2*window_size).

    Returns
    -------
    mat_signal : numpy matrix
        a numpy matrix with the mean signal around all the positions of interest.
        The shape of mat_signal is (2*window_size)x(2*window_size).
    nb_positions : int
        the number of positions averaged
    """
    print("\n" + "-" * 20 + "\ngetting the agglomerated signal\n")
    mat_signal = np.zeros((window_size * 2 + 1, window_size * 2 + 1))
    mat_signal, no_positions = pile(
        mat_signal, matrix, matrices_limits, binning, positions, window_size
    )
    return mat_signal, no_positions

    # # Get the good variables
    # if no_arms:
    #     current_matrix = np.copy(matrices[chrm]["mat"])
    #     current_correct_indices = set(matrices[chrm]["indices"])
    #     current_chrm_right_border = current_matrix.shape[0]
    #     borders_of_interest = df[df["chrm"] == chrm]
    #     nb_positions += len(borders_of_interest)
    #     mat_agglo1, mat_occ1 = compute_signal(
    #         mat_agglo1,
    #         mat_occ1,
    #         current_matrix,
    #         current_correct_indices,
    #         current_chrm_right_border,
    #         borders_of_interest,
    #         window,
    #     )
    # else:
    #     for side in ["L", "R"]:
    #         current_matrix = np.copy(matrices[chrm]["mat" + side])
    #         current_correct_indices = set(matrices[chrm]["indices" + side])
    #         current_centro = matrices[chrm]["centro"]
    #         current_chrm_right_border = current_matrix.shape[0]
    #         if side is "L":
    #             borders_of_interest = df.loc[
    #                 (df["chrm"] == chrm) & (df["pos"] < current_centro)
    #             ]
    #         else:
    #             borders_of_interest = df.loc[
    #                 (df["chrm"] == chrm) & (df["pos"] >= current_centro)
    #             ]
    #             borders_of_interest["pos"] = borders_of_interest["pos"] - current_centro
    #             borders_of_interest["pos2"] = (
    #                 borders_of_interest["pos2"] - current_centro
    #             )
    #         nb_positions += len(borders_of_interest)
    #         mat_agglo1, mat_occ1 = compute_signal(
    #             mat_agglo1,
    #             mat_occ1,
    #             current_matrix,
    #             current_correct_indices,
    #             current_chrm_right_border,
    #             borders_of_interest,
    #             window,
    #         )

    # return np.divide(mat_agglo1, mat_occ1), nb_positions


# def get_null_model_mean(matrices, list_chr, df, window, nb_iter, no_arms):


def get_null_model_mean(matrices, list_chr, df, window, nb_iter, no_arms):
    """Generates null model."""
    print("\n" + "-" * 20 + "\ngenerating the null models\n")
    mat_agglo2 = np.zeros((window * 2 + 1, window * 2 + 1))
    mat_agglo_aux = np.zeros((window * 2 + 1, window * 2 + 1))
    mat_occ_aux = np.zeros((window * 2 + 1, window * 2 + 1))
    # The computation of a null model starts here
    for chrm in list_chr:
        # Get the good variables
        if no_arms:
            current_matrix = np.copy(matrices[chrm]["mat"])
            current_correct_indices = set(matrices[chrm]["indices"])
            current_chrm_right_border = current_matrix.shape[0]
            borders_of_interest = df[df["chrm"] == chrm]
            mat_agglo_aux, mat_occ_aux = compute_signal(
                mat_agglo_aux,
                mat_occ_aux,
                current_matrix,
                current_correct_indices,
                current_chrm_right_border,
                borders_of_interest,
                window,
                random=True,
            )
        else:
            for side in ["L", "R"]:
                current_matrix = matrices[chrm]["mat" + side]
                current_correct_indices = set(matrices[chrm]["indices" + side])
                current_centro = matrices[chrm]["centro"]
                current_chrm_right_border = current_matrix.shape[0]
                if side is "L":
                    borders_of_interest = df.loc[
                        (df["chrm"] == chrm) & (df["pos"] < current_centro)
                    ]
                else:
                    borders_of_interest = df.loc[
                        (df["chrm"] == chrm) & (df["pos"] >= current_centro)
                    ]
                    borders_of_interest["pos"] = (
                        borders_of_interest["pos"] - current_centro
                    )
                    borders_of_interest["pos2"] = (
                        borders_of_interest["pos2"] - current_centro
                    )
                mat_agglo_aux, mat_occ_aux = compute_signal(
                    mat_agglo_aux,
                    mat_occ_aux,
                    current_matrix,
                    current_correct_indices,
                    current_chrm_right_border,
                    borders_of_interest,
                    window,
                    random=True,
                )

                # Compute the signal
                ## for each border, choose a random central bin, take the signal in the bins around the central bin and add it to mat_agglo_aux.
                ## since some bins don't exist, keep the count of the number of times each bin around the center of mat_agglo_aux
                ## is incremented in mat_occ_aux. mat_occ_aux will be used for normalization of mat_agglo_aux.
                # for index, row in borders_of_interest.iterrows():
                #     diff = row["pos2"] - row["pos"]
                #     site1 = (
                #         rando.randint(0, current_chrm_right_border - diff)
                #         if (current_chrm_right_border - diff > 0)
                #         else rando.randint(0, current_chrm_right_border)
                #     )
                #     site2 = site1 + diff
                #     for i in range(site1 - window, site1 + window + 1):
                #         for j in range(site2 - window, site2 + window + 1):
                #             if (
                #                 i >= 0
                #                 and j >= 0
                #                 and i < current_chrm_right_border
                #                 and j < current_chrm_right_border
                #                 and (i in current_correct_indices)
                #                 and (j in current_correct_indices)
                #             ):
                #                 mat_agglo_aux[
                #                     i - site1 + window, j - site2 + window
                #                 ] = (
                #                     mat_agglo_aux[
                #                         i - site1 + window, j - site2 + window
                #                     ]
                #                     + current_matrix[i, j]
                #                     + current_matrix[j, i]
                #                 )
                #                 mat_occ_aux[i - site1 + window, j - site2 + window] = (
                #                     mat_occ_aux[i - site1 + window, j - site2 + window]
                #                     + 2.0
                #                 )
        # The computation of the null model (mat_agglo_aux) is done. Increment mat_agglo2 with it. In the end, matt_agglo2 will be averaged
    #     mat_agglo_aux = mat_agglo_aux / mat_occ_aux
    #     mat_agglo_aux = np.nan_to_num(mat_agglo_aux)
    #     # print("l554: {mat_agglo_aux}".format(mat_agglo_aux= mat_agglo_aux))
    #     # plt.imshow(mat_agglo_aux)
    #     # plt.show()
    #     mat_agglo2 = mat_agglo2 + mat_agglo_aux
    #     mat_agglo2 = np.nan_to_num(mat_agglo2)

    # # plt.imshow(mat_agglo2)
    # # plt.show()
    # mat_agglo2 = mat_agglo2 / nb_iter
    # return mat_agglo2


def pile(
    mat_signal, matrix, matrices_limits, binning, positions, window_size, random=False
):
    """Compute the signal for each border, take the signal in the bins around
    of the border and add it to mat_agglo1. Since some bins don't exist, keep
    the count of the number of times each bin around the center of mat_agglo1
    is incremented in mat_occ. mat_occ1 will be used for normalization of
    mat_agglo1.

    Parameters
    ----------numpy array
        Array keeping count of the number of values summed for each pixel
    current_matrix: numpy array
        Array corresponding to the the contact map
    current_correct_indices: set
        a set of int with all the indices of the bins which are not undercovered and can be used
    current_chrm_right_border: int
        the bin corresponding to the right boder of the useful contact map. Can be the shape of the contact map or the centromer if
        the positions inter-arms are excluded.
    borders_of_interest: pandas DataFrame
        a dataframe with all the positions to pileup on the current_matrix.
    window: int
        the size of the window to slice around the positions of interest (in pixels)
    random: bool
        indicate if the sites to chose are the one provided by borders_of_interest or if random sites must be picked randomly
        (when generating the null model)

        Returns
        -------
        mat_agglo : numpy array
            A new version of mat_agglo updated with the signal found on the given contact map was added
        mat_occ : numpy array
            A new version of mat_occ updated whith the number of values addeded to each pixel
    """
    list_results = []

    for chrm in matrices_limits.keys():
        for tup in matrices_limits[chrm]:
            start = tup[0]
            end = tup[1]
            submatrix = matrix[start:end, start:end]
            submatrix_length = submatrix.shape[0]
            subpositions = positions[
                (positions["chr1"] == chrm)
                & (positions["pos1"] >= window_size)
                & (positions["pos2"] >= window_size)
                & (positions["pos1"] < submatrix_length - window_size)
                & (positions["pos2"] < submatrix_length - window_size)
            ]

            for index, row in subpositions.iterrows():
                if not random:
                    site1 = int(row["pos1"])
                    site2 = int(row["pos2"])
                else:
                    diff = row["pos2"] - row["pos"]
                    site1 = (
                        rando.randint(0, current_chrm_right_border - diff)
                        if (current_chrm_right_border - diff > 0)
                        else rando.randint(0, current_chrm_right_border)
                    )
                    site2 = site1 + diff
                list_results.append(
                    submatrix[
                        site1 - window_size : site2 + window_size + 1,
                        site1 - window_size : site2 + window_size + 1,
                    ].toarray()
                )
    return np.nanmean(np.array(list_results), axis=0)

    #         for i in range(site1 - window, site1 + window + 1):
    #             for j in range(site2 - window, site2 + window + 1):
    #                 if (
    #                     i >= 0
    #                     and j >= 0
    #                     and i < current_chrm_right_border
    #                     and j < current_chrm_right_border
    #                     and (i in current_correct_indices)
    #                     and (j in current_correct_indices)
    #                 ):
    #                     mat_agglo[i - site1 + window, j - site2 + window] = (
    #                         mat_agglo[i - site1 + window, j - site2 + window]
    #                         + current_matrix[i, j]
    #                         + current_matrix[j, i]
    #                     )
    #                     mat_occ[i - site1 + window, j - site2 + window] = (
    #                         mat_occ[i - site1 + window, j - site2 + window] + 2.0
    #                     )
    # return mat_agglo, mat_occ


# def get_agglomerated_signal_median(matrices, list_chr, df, window):
#     """Gets a list of matricse and of positions of interest and returns 1 matrix corresponding to
#     the agglomeration of the signal around these positions."""
#     print("\n" + "-" * 20 + "\ngetting the agglomerated signal\n")
#     # ~ mat_agglo1 = np.zeros((window*2+1,window*2+1))
#     mat_agglo1 = np.zeros(((2 * window + 1) ** 2, df.shape[0]))
#     mat_agglo1 = mat_agglo1.astype("float")
#     mat_agglo1[mat_agglo1 == 0] = np.nan
#     nb_positions = 0
#     column = -1

#     for chrm in list_chr:
#         # Get the good variables
#         for side in ["L", "R"]:
#             current_matrix = np.copy(matrices[chrm]["mat" + side])
#             current_correct_indices = set(matrices[chrm]["indices" + side])
#             current_centro = matrices[chrm]["centro"]
#             current_chrm_right_border = current_matrix.shape[0]
#             if side is "L":
#                 borders_of_interest = df.loc[
#                     (df["chrm"] == chrm) & (df["pos"] < current_centro)
#                 ]
#             else:
#                 borders_of_interest = df.loc[
#                     (df["chrm"] == chrm) & (df["pos"] >= current_centro)
#                 ]
#                 borders_of_interest["pos"] = borders_of_interest["pos"] - current_centro
#                 borders_of_interest["pos2"] = (
#                     borders_of_interest["pos2"] - current_centro
#                 )
#             nb_positions += borders_of_interest.shape[0]

#             # Compute the signal
#             ## for each border, take the signal in the bins around of the border and add it to mat_agglo1.
#             ## since some bins don't exist, keep the count of the number of times each bin around the center of mat_agglo1
#             ## is incremented in mat_occ. mat_occ1 will be used for normalization of mat_agglo1
#             for index, row in borders_of_interest.iterrows():
#                 site1 = int(row["pos"])
#                 site2 = int(row["pos2"])
#                 column += 1
#                 for i in range(site1 - window, site1 + window + 1):
#                     for j in range(site2 - window, site2 + window + 1):
#                         if (
#                             i >= 0
#                             and j >= 0
#                             and i < current_chrm_right_border
#                             and j < current_chrm_right_border
#                         ):
#                             # ~ and (i in current_correct_indices) \
#                             # ~ and (j in current_correct_indices):
#                             indice = (i - site1 + window) * (2 * window + 1) + (
#                                 j - site2 + window
#                             )
#                             mat_agglo1[indice, column] = current_matrix[i, j]

#     mat_agglo1v2 = [np.nanmedian(mat_agglo1[i]) for i in range(mat_agglo1.shape[0])]

#     mat_agglo1v2 = np.array(mat_agglo1v2).reshape((2 * window + 1, 2 * window + 1))

#     return mat_agglo1v2, nb_positions

# def get_null_model_median(matrices, list_chr, df, window, nb_iter=10):
#     """Generates null model"""
#     print("\n" + "-" * 20 + "\ngenerating the null models\n")
#     mat_agglo2 = np.zeros((window * 2 + 1, window * 2 + 1))

#     # Do several (nb_iter) null models and average them for a better accuracy(10 is generally good).
#     for r in range(nb_iter):
#         mat_aux = np.zeros(((2 * window + 1) ** 2, df.shape[0]))
#         mat_aux = mat_aux.astype("float")
#         mat_aux[mat_aux == 0] = np.nan
#         column = -1

#         # ~ print("{0}/{1}".format(r, nb_iter))
#         for chrm in list_chr:
#             # Get the good variables
#             for side in ["L", "R"]:
#                 current_matrix = np.copy(matrices[chrm]["mat" + side])
#                 current_correct_indices = set(matrices[chrm]["indices" + side])
#                 current_centro = matrices[chrm]["centro"]
#                 current_chrm_right_border = current_matrix.shape[0]
#                 if side is "L":
#                     borders_of_interest = df.loc[
#                         (df["chrm"] == chrm) & (df["pos"] < current_centro)
#                     ]
#                 else:
#                     borders_of_interest = df.loc[
#                         (df["chrm"] == chrm) & (df["pos"] >= current_centro)
#                     ]
#                     borders_of_interest["pos"] = (
#                         borders_of_interest["pos"] - current_centro
#                     )
#                     borders_of_interest["pos2"] = (
#                         borders_of_interest["pos2"] - current_centro
#                     )

#                 # Compute the signal
#                 ## for each border, take the signal in the bins around of the border and add it to mat_agglo1.
#                 ## since some bins don't exist, keep the count of the number of times each bin around the center of mat_agglo1
#                 ## is incremented in mat_occ. mat_occ1 will be used for normalization of mat_agglo1
#                 for index, row in borders_of_interest.iterrows():
#                     diff = row["pos2"] - row["pos"]
#                     site1 = (
#                         rando.randint(0, current_chrm_right_border - diff)
#                         if (current_chrm_right_border - diff > 0)
#                         else rando.randint(0, current_chrm_right_border)
#                     )
#                     site2 = site1 + diff
#                     column += 1
#                     for i in range(site1 - window, site1 + window + 1):
#                         for j in range(site2 - window, site2 + window + 1):
#                             if (
#                                 i >= 0
#                                 and j >= 0
#                                 and i < current_chrm_right_border
#                                 and j < current_chrm_right_border
#                             ):
#                                 # ~ and (i in current_correct_indices) \
#                                 # ~ and (j in current_correct_indices):
#                                 indice = (i - site1 + window) * (2 * window + 1) + (
#                                     j - site2 + window
#                                 )
#                                 mat_aux[indice, column] = current_matrix[i, j]

#         mat_aux2 = [np.nanmedian(mat_aux[i]) for i in range(mat_aux.shape[0])]
#         mat_aux2 = np.array(mat_aux2).reshape((2 * window + 1, 2 * window + 1))
#         mat_agglo2 += mat_aux2

#     return mat_agglo2 / nb_iter


# ~ def detrend_matrix(mn):
# ~ dist = distance_law_human(mn)
# ~ n1 = mn.shape[0]
# ~ mat_dist =  np.zeros((n1, n1))
# ~ for i in range(0,n1) :
# ~ for j in range(0,n1) :
# ~ mat_dist[i,j] =  dist[abs(j-i)]
# ~ mat_detrend = mn / mat_dist
# ~ mat_detrend[np.isnan(mat_detrend)] = 1.0    # why put a NaN to 1 ?
# ~ return mat_detrend

# ~ def distance_law_human(A):
# ~ n1 = A.shape[0]
# ~ dist = np.zeros((n1, 1))
# ~ for nw in range(n1):  # scales
# ~ somme = []
# ~ for i in range(n1):
# ~ j = i - nw
# ~ if (j >= 0) and (j < n1):
# ~ somme.append(A[i, j])
# ~ dist[nw] = np.mean(somme)
# ~ return dist

# =============================================
# Scoring the results
# =============================================
# def compute_score(result_signal, scale, margin=1):
#     """Compute the strength of the loops in a list of ratio matrices by taking the mean value
#     of the central picexls ans dividing it by the mean values of the upper and right borders
# # Scoring the results
# # =============================================
def compute_score(result_signal, scale, margin=1):
    """Compute the strength of the loops in a list of ratio matrices by taking
    the mean value of the central picexls ans dividing it by the mean values of
    the upper and right borders wich serve as an approximate for the
    background.

    Parameters
    ----------
        the number of pixels to use to compute loop_value and background_value

    Returns
    -------
    loops_strengths: list
        a list of int corresponding to the strengh of loops for each ratio matrix
    """

    loops_strengths = []
    for matrix in result_signal:
        if scale == "logratio":
            matrix = 2 ** matrix
        n = matrix.shape[0]
        submatrix_loop = matrix[
            n // 2 - margin : n // 2 + margin + 1, n // 2 - margin : n // 2 + margin + 1
        ]
        submatrix_background = matrix[: n // 2, n // 2 + margin :]
        strength = np.median(submatrix_loop) / np.median(submatrix_background)
        # submatrix = matrix[
        #         int(n / 2 - margin) : int(n / 2 + margin + 1),
        #         int(n / 2 - margin) : int(n / 2 + margin + 1),
        #     ]
        # weights = [1/(d+1) for d in range(margin, 0, -1)] + [1] + [1/(d+1) for d in range(1, margin+1)]
        # a, b = np.meshgrid(weights, weights, sparse=True)
        # weights = (a + b) / 2
        # loop_value = np.average(submatrix, weights=weights)
        # background_value = (
        #     np.median(matrix[: 2 * margin, :])
        #     + np.median(matrix[2 * margin :, -2 * margin :])
        # ) / 2
        loops_strengths.append(strength)  # / background_value)
    return loops_strengths


# =============================================
# Plotting the results
# =============================================
def plot_agglo(
    matrices,
    positions_count,
    titles,
    suptitle,
    window,
    cs,
    BIN,
    masks,
    pileup,
    m_vmin,
    m_vmax,
    scale,
    loops_strengths,
):
    # Define colormap
    if not pileup:
        # c = mcolors.ColorConverter().to_rgb
        # my_cmap = make_colormap([c('mediumblue'), c('lavender'), 0.4, c('lavender'), c('white'), 0.5, c('white'), c('mistyrose'), 0.6, c('mistyrose'), c('red')])
        my_cmap = copy(plt.cm.seismic)
    else:
        my_cmap = copy(plt.cm.afmhot_r)
    my_cmap.set_bad("grey", 1.0)

    # Get number of heatmaps
    nb_col = len(matrices)

    # Draw figure
    fig = plt.figure(figsize=(5 * nb_col, 5))
    gs = gridspec.GridSpec(
        1, nb_col, wspace=0.5
    )  # , hspace=0.05, height_ratios=[0.5, 9.5, 1])#, width_ratios=[1, 10])
    ##main ax
    for i in range(nb_col):
        # add a mask if needed
        logratio = matrices[i]
        strength = np.round(loops_strengths[i], 2)
        size = logratio.shape[0]
        # ~ l1 = logratio[:size+1,:size+1]
        # ~ l2 = logratio[:size+1,size:]
        # ~ l3 = logratio[size:,:size+1]
        # ~ l4 = logratio[size:,size:]
        # ~ print(l1.shape, l2.shape, l3.shape, l4.shape)
        # ~ total = l4 + np.fliplr(l3) + np.flipud(l2) + np.flipud(np.fliplr(l1))
        # ~ logratio = total / 4

        # If the argument --mask was passed, add masks to the lower left corner
        if masks:
            logratio = make_mask(logratio, masks[i])

        # Add a subplot for the ratio matrix
        ax = plt.subplot(gs[i])
        # Plot the ratio matrix with the good graphical parameters
        if not pileup:
            if scale == "logratio":
                im = ax.imshow(
                    logratio, vmin=-1 * cs, vmax=cs, cmap=my_cmap, interpolation="none"
                )
                plt.colorbar(
                    im,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.5,
                    ticks=[-1 * cs, 0, cs],
                )
            else:
                im = ax.imshow(
                    logratio,
                    vmin=1 - cs,
                    vmax=1 + cs,
                    cmap=my_cmap,
                    interpolation="none",
                )
                plt.colorbar(
                    im,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.5,
                    ticks=[1 - cs, 1, 1 + cs],
                )
        else:
            im = ax.imshow(
                logratio ** cs,
                cmap=my_cmap,
                interpolation="none",
                vmin=m_vmin,
                vmax=m_vmax,
            )
            plt.colorbar(
                im, ax=ax, orientation="horizontal", shrink=0.5, ticks=[0.0, 0.8]
            )

        # Ticks and labels
        ## the size of the window w
        tick_lbls = [
            "- " + str(int(window * BIN / 1000)) + "kb",
            str(0) + "kb",
            "+ " + str(int(window * BIN / 1000)) + "kb",
        ]
        tick_locs = [0, size / 2, size]
        ax.set_xticks(tick_locs)
        ax.set_yticks(tick_locs)
        ax.set_xticklabels(tick_lbls, {"size": "large"})
        plt.tick_params(axis="y", which="both", direction="out", labelleft=False)
        plt.tick_params(axis="x", direction="out")
        # ~ plt.xlabel("Number of agglomerated images "+str(dfs[i].shape[0]))
        ## the number of positions agglomerated for this ratio matrix
        plt.xlabel(str(positions_count[i]))
        ## the title of the matrix
        if titles is not None:
            ax.set_title(titles[i])
        if suptitle is not None:
            plt.suptitle(suptitle.replace("_", " "), fontsize=14)
        ## the strength of the loop
        plt.text(2, 4, strength, {"size": "x-large", "weight": "bold"})

    return fig


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


def adjust_masks_length(masks, a, b):
    for i in range(a, b):
        masks.append(0)
    return masks


def make_mask(logratio, ratio_mask):
    dim = logratio.shape[0]
    my_mask = np.zeros((dim, dim))
    for i in range(dim - 1, dim - int(ratio_mask * dim), -1):
        for j in range(0, int(ratio_mask * dim) - (dim - i - 1)):
            my_mask[i, j] = 1
    mlogratio = ma.array(logratio, mask=my_mask)
    return mlogratio


# =============================================
# Saving the results
# =============================================


def save_txt_matrices(result_signal, outimg, titles):
    os.system("mkdir -p {0}".format(outimg))
    for i, matrix in enumerate(result_signal):
        name = pathlib.Path(outimg) / "{0}.txt".format(titles[i])
        np.savetxt(name, matrix)


def save_my_fig(my_fig, outimg):
    outimg1 = outimg + ".png"
    my_fig.savefig(outimg1, dpi=300)
    outimg2 = outimg1.replace(".png", ".pdf")
    my_fig.savefig(outimg2, dpi=200)
    print("\nSaved images as {0} and {1}".format(outimg1, outimg2))


# ~ def save_my_data(result_signal, positions_count, outimg):
# ~ np.save(outimg+'.npy', result_signal)
# ~ np.save(outimg+'.npy', result_signal)
# ~ with open(outimg+'_positions.txt', 'wb') as fp:
# ~ pickle.dump(positions_count, fp)

# =======================================================================
# MAIN
# =======================================================================
def main():
    print("\n" + "-" * 40 + "\nBeginning of program\n")

    # Get arguments
    args = parse_args()
    ## positional arguments
    BIN = args.bins
    borders_files = args.borders.split(",")
    contact_maps_folders = args.folder.split(",")

    ## optional arguments
    centromers_file = os.path.join(args.centromers) if args.centromers else None
    cs = args.color_scale
    list_chr = args.list_chr.split(",")
    masks = (
        [float(elt) for elt in args.masks.split(",")]
        if args.masks is not None
        else args.masks
    )
    metric = args.metric
    mode = args.mode
    nb_iter = args.nb_iter  # number of random agglomeration to perform for null model
    nodisp = args.no_disp
    no_arms = args.no_arms
    outimg = args.outimg
    pileup = args.pileup
    titles = args.titles.split(",") if args.titles is not None else args.titles
    scale = args.scale
    m_vmin = args.vmin
    m_vmax = args.vmax
    window = args.window  # number of bins around the sites of interest

    # outimg = outimg + "_{0}kb_w{1}_cs{2}".format(int(BIN/1000), window, cs)

    # Load centromers dataframe
    if not no_arms:
        if not centromers_file:
            print("ERROR: Indicate a centromer file (-cen) or precise --no_arms.")
            sys.exit(3)
        else:
            try:
                centromers = pd.read_csv(
                    centromers_file, sep=" ", names=["chrm", "centro"], usecols=range(2)
                )
            except:
                print("ERROR: The centromers file does not exist")
                sys.exit(3)
        centromers.centro = centromers.centro // BIN
        print(centromers)
        print("\n")
    else:
        centromers = None

    # Adjust the mask list
    if masks and len(masks) < len(borders_files):
        adjust_masks_length(masks, len(masks), len(borders_files))
    suptitle = args.suptitle

    # Sanity checks
    ## as many borders files as titles ?
    assert titles is None or (
        len(borders_files) == len(titles)
    ), "There is not 1 title by heatmap. Please give a title for each borders file given.\n"
    ## do the borders files exist ?
    for borders_file in borders_files:
        if not os.path.isfile(borders_file):
            sys.stderr.write(
                "ERROR: The file {0} does not exist.\n".format(borders_file)
            )
            sys.exit(1)
    ## each chromosome of chromosome list is in the centromers list ?
    if not no_arms:
        for chrm in list_chr:
            if chrm not in centromers.chrm.tolist():
                sys.stderr.write(
                    "ERROR: There are chromosome in list_chr which are not in centromers_file. Check: {0}.\n".format(
                        list_chr
                    )
                )
                sys.exit(2)

    # Get agglomarated signal matrix, null model, make the ratio and plot the resul
    # ~ if mode != 'both':
    result_signal, positions_count, my_fig = get_agglomerated_plot(
        list_chr,
        contact_maps_folders,
        borders_files,
        BIN,
        centromers,
        window,
        nb_iter,
        cs,
        titles,
        suptitle,
        no_arms,
        outimg,
        pileup,
        masks,
        metric,
        m_vmin,
        m_vmax,
        scale,
    )
    if mode in ["loop", "loops"]:
        suff = "_loops"
    elif mode in ["border", "borders"]:
        suff = "_borders"
    else:
        suff = ""
    save_my_fig(my_fig, outimg + suff)

    # ~ else:
    # ~ result_signal, positions_count, my_fig = get_agglomerated_plot(list_chr, contact_maps_folder, borders_files, BIN, centromers, window, nb_iter, cs, titles, suptitle, outimg, masks, 'borders')
    # ~ save_my_fig(my_fig, outimg+'_borders')
    # ~ if not nodisp: plt.show()
    # ~ result_signal, positions_count, my_fig = get_agglomerated_plot(list_chr, contact_maps_folder, borders_files, BIN, centromers, window, nb_iter, cs, titles, suptitle, outimg, masks, 'loops')
    # ~ save_my_fig(my_fig, outimg+'_loops')
    if not nodisp:
        plt.show()

    print("\nEnd of program\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
