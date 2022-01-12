
import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
import matplotlib.pyplot as plt
import itertools
from helper import *


class Automata:
    def __init__(self, trans_hl, trans_lh):
        self.transition_probabilities = np.zeros([2, 2])
        self.initialize_transition_probabilities(trans_hl, trans_lh)

    def initialize_transition_probabilities(self, trans_hl, trans_lh):
        self.transition_probabilities[0, 0] = 1 - trans_hl
        self.transition_probabilities[0, 1] = trans_hl
        self.transition_probabilities[1, 1] = 1 - trans_lh
        self.transition_probabilities[1, 0] = trans_lh

        with np.errstate(divide='ignore'):
            self.transition_probabilities = np.log(self.transition_probabilities)


def forward_alg_log_vectorized(seq, trans_hl, trans_lh, emission_table):
    automata = Automata(trans_hl, trans_lh)
    forward_table = np.zeros([2, seq.shape[1]])
    with np.errstate(divide='ignore'):
        forward_table = np.log(forward_table)
    forward_table[:, 0] = emission_table[:, 0]
    for cur_seq_index in range(1, seq.shape[1]):
        prev_f_col = forward_table[:, cur_seq_index - 1]
        prev_f_col_mat = np.tile(prev_f_col, (2, 1)).T
        element_multiply = prev_f_col_mat + automata.transition_probabilities
        dot_product = logsumexp(element_multiply, axis=0)
        log_emission_vector = emission_table[:, cur_seq_index]
        forward_table[:, cur_seq_index] = dot_product + log_emission_vector
    return forward_table, logsumexp([forward_table[0, -1], forward_table[1, -1]])


def backward_alg_log_vectorized(seq, trans_hl, trans_lh, emission_table):
    automata = Automata(trans_hl, trans_lh)
    backward_table = np.zeros([2, seq.shape[1]])
    with np.errstate(divide='ignore'):
        backward_table = np.log(backward_table)
    backward_table[:, -1] = [0, 0]
    for cur_seq_index in range(seq.shape[1]-2, -1, -1):
        future_column = backward_table[:, cur_seq_index + 1]
        log_emission_vector = emission_table[:, cur_seq_index + 1]
        emission_plus_future = log_emission_vector + future_column
        emission_future_tiled = np.tile(emission_plus_future, (2, 1))
        res = emission_future_tiled + automata.transition_probabilities
        backward_table[:, cur_seq_index] = logsumexp(res, axis=1)

    ll_backward = logsumexp(emission_table[:, 0] + backward_table[:, 0])
    return backward_table, ll_backward


def calc_parameters(ph, pl, hl, lh, fasta, ll_history, plot_mode_only=False):
    # based on (hl, lh) we can compute the probability for each cell to emit its proportion from the H, L
    # ph, pl is the probability for C (num of mathylated)
    emission_table = np.zeros([2, fasta.shape[1]])
    num_C, total_reads_vec = fasta[0, :], fasta[1, :]
    num_T = total_reads_vec - num_C
    emission_table[0, :] = binom.pmf(num_C, total_reads_vec, ph)
    emission_table[1, :] = binom.pmf(num_C, total_reads_vec, pl)
    with np.errstate(divide='ignore'):
        emission_table = np.log(emission_table)
    forward_table, fasta_llf = forward_alg_log_vectorized(fasta, hl, lh, emission_table)
    backward_table, fasta_llb = backward_alg_log_vectorized(fasta, hl, lh, emission_table)
    f_plus_b = np.exp(forward_table + backward_table - fasta_llf)  # shape = [2, len(seq)]

    if plot_mode_only:
        print('final transition table: (order of rows/ columns is [HIGH, LOW])\n', np.array([[1-hl, hl], [lh, 1-lh]]))

        print('probability of mathylation in HIGH state', ph)
        print('probability of mathylation in LOW state', pl)
        plot_results_2_states(fasta, ph, pl, f_plus_b)
        return

    ll_history += [fasta_llf]
    # phl is probability to transition from H to L
    est_phl_up = np.sum(np.exp(forward_table[0, :-1] + np.log(hl) + emission_table[1, 1:] + backward_table[1, 1:] - fasta_llf))
    est_phl = est_phl_up / np.sum(np.exp(forward_table[0, :-1] + backward_table[0, :-1] - fasta_llf))
    # plh is probability to transition from L to H
    est_plh_up = np.sum(np.exp(forward_table[1, :-1] + np.log(lh) + emission_table[0, 1:] + backward_table[0, 1:] - fasta_llf))
    est_plh = est_plh_up / np.sum(np.exp(forward_table[1, :-1] + backward_table[1, :-1] - fasta_llf))

    p_C = np.multiply(num_C, f_plus_b)
    p_T = np.multiply(num_T, f_plus_b)
    est_ph = np.sum(p_C[0, :]) / (np.sum(p_C[0, :]) + np.sum(p_T[0, :]))
    est_pl = np.sum(p_C[1, :]) / (np.sum(p_C[1, :]) + np.sum(p_T[1, :]))
    print('current log likelihood', fasta_llf)
    return est_ph, est_pl, est_phl, est_plh


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', help='File path with name of the beta file to be used as input',
                        default="Prostate-Epithelial-Z000000S3.small_range.beta")
    parser.add_argument('--ph', type=float, help='Initial guess for the "H" hidden state binomial distibrution parameter(e.g. 0.9)',
                        default=0.85)
    parser.add_argument('--pl', type=float, help='Initial guess for the "L" hidden state binomial distibrution parameter(e.g. 0.1)',
                        default=0.2)
    parser.add_argument('--hl', type=float, help='Initial guess for the transition probability from H to L (e.g. 0.01)',
                        default=0.05)
    parser.add_argument('--lh', type=float, help='Initial guess for the transition probability from L to H (e.g. 0.9)',
                        default=0.1)
    parser.add_argument('--start', type=float, help='index of the CpG site from which to start (e.g. 1254)',
                        default=50000)
    parser.add_argument('--end', type=float, help='index of the CpG site from which to end (e.g. 10000)',
                        default=60000)
    parser.add_argument('--convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                             ' (e.g. 0.1)', default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    filename = args.beta
    # a small portion of the Prostata of size 200,000
    # small_filename = 'Prostate-Epithelial-Z000000S3.small_range.beta'

    fasta = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
    start_ind = args.start
    end_ind = args.end
    fasta = fasta[start_ind:end_ind, :]  # or any other random range

    fasta = eliminate_zeros_add_one(fasta)
    ph, pl, hl, lh = args.ph, args.pl, args.hl, args.lh
    convergenceThr = args.convergenceThr

    # run Baum-Welch
    ll_history = []
    while len(ll_history) < 2 or abs(ll_history[-1] - ll_history[-2]) > convergenceThr:
        ph, pl, hl, lh = calc_parameters(ph, pl, hl, lh, fasta.T, ll_history)

    ph, pl, hl, lh = calc_parameters(ph, pl, hl, lh, fasta.T, ll_history, plot_mode_only=False)

    print("Hidden state H success param: " + str(ph))
    print("Hidden state L success param: " + str(pl))
    print("Transition probability from H to L: " + str(hl))
    print("Transition probability from L to H: " + str(lh))

