
import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
import matplotlib.pyplot as plt
from helper import *


def forward_alg_log_vectorized(fasta, transitions, emission_table):
    forward_table = np.zeros([3, fasta.shape[1]])
    with np.errstate(divide='ignore'):
        forward_table = np.log(forward_table)

    forward_table[:, 0] = emission_table[:, 0]
    for cur_seq_index in range(1, fasta.shape[1]):
        prev_f_col = forward_table[:, [cur_seq_index - 1]]
        prev_f_col_mat = np.tile(prev_f_col, (1, 3))
        element_multiply = prev_f_col_mat + transitions
        dot_product = logsumexp(element_multiply, axis=0)
        log_emission_vector = emission_table[:, cur_seq_index]
        forward_table[:, cur_seq_index] = dot_product + log_emission_vector
    return forward_table, logsumexp([forward_table[0, -1], forward_table[1, -1]])


def backward_alg_log_vectorized(fasta, transitions, emission_table):
    backward_table = np.zeros([3, fasta.shape[1]])
    with np.errstate(divide='ignore'):
        backward_table = np.log(backward_table)
    backward_table[:, -1] = [0, 0, 0]
    for cur_seq_index in range(fasta.shape[1]-2, -1, -1):
        future_column = backward_table[:, cur_seq_index + 1]
        log_emission_vector = emission_table[:, cur_seq_index + 1]
        emission_plus_future = log_emission_vector + future_column
        emission_future_tiled = np.tile(emission_plus_future, (3, 1))
        res = emission_future_tiled + transitions
        backward_table[:, cur_seq_index] = logsumexp(res, axis=1)
    ll_backward = logsumexp(emission_table[:, 0] + backward_table[:, 0])
    return backward_table, ll_backward


def calc_parameters(transitions, ph, fasta, ll_history, plot_mode_only=False):
    emission_table = np.zeros([3, fasta.shape[1]])
    num_C, total_reads_vec = fasta[0, :], fasta[1, :]
    num_T = total_reads_vec - num_C

    emission_table[0, :] = binom.pmf(num_C, total_reads_vec, ph[0])
    emission_table[1, :] = binom.pmf(num_C, total_reads_vec, ph[1])
    emission_table[2, :] = binom.pmf(num_C, total_reads_vec, ph[2])
    with np.errstate(divide='ignore'):
        emission_table = np.log(emission_table)
        transitions = np.log(transitions)
    forward_table, fasta_llf = forward_alg_log_vectorized(fasta, transitions, emission_table)
    backward_table, fasta_llb = backward_alg_log_vectorized(fasta, transitions, emission_table)
    ll_history += [fasta_llf]
    likelihood = np.exp(forward_table + backward_table - fasta_llf)

    if plot_mode_only:
        print('final transition table: (order of rows/ columns is [HIGH, LOW, OTHER])\n', np.exp(transitions))

        print('probability of mathylation in HIGH state', ph[0])
        print('probability of mathylation in LOW state', ph[1])
        print('probability of mathylation in OTHER state', ph[2])
        plot_results_3_states(fasta, ph, likelihood)
        return

    # ESTIMATE TRANSITIONS
    est_transitions = np.zeros([3, 3])
    dividers = np.sum(likelihood[:, :-1], axis=1)  # sum all columns in row
    for from_state in range(3):
        for to_state in range(3):
            est_transitions[from_state, to_state] = np.sum(np.exp(forward_table[from_state, :-1] + transitions[from_state, to_state] +
                                                                  emission_table[to_state, 1:] + backward_table[to_state, 1:] - fasta_llf))
            est_transitions[from_state, to_state] /= dividers[from_state]

    # ESTIMATE BERNOULLI PARAMETER
    p_C = np.multiply(num_C, likelihood)
    p_T = np.multiply(num_T, likelihood)
    est_ph = np.sum(p_C, axis=1) / (np.sum(p_C, axis=1) + np.sum(p_T, axis=1))
    print('current log likelihood', fasta_llf)
    return est_transitions, est_ph


if __name__ == '__main__':
    start = 50000
    end = 60000
    filename = 'Prostate-Epithelial-Z000000S3.beta'
    # a small portion of the Prostata of size 200,000
    small_filename = 'Prostate-Epithelial-Z000000S3.small_range.beta'
    fasta = np.fromfile(small_filename, dtype=np.uint8).reshape((-1, 2))
    fasta = fasta[start:end, :]  # or any other random range

    fasta = eliminate_zeros_add_one(fasta)
    transitions = np.array([[0.88, 0.04, 0.08], [0.08, 0.8, 0.1],  [0.75, 0.1, 0.15]])  # [[hth, htl, hto], [lth, ltl, lto], [oth, otl, oto]]
    ph = np.array([0.88, 0.08, 0.2])  # HIGH, LOW, OTHER
    convergenceThr = 0.5

    # run Baum-Welch
    ll_history = []
    while len(ll_history) < 2 or abs(ll_history[-1] - ll_history[-2]) > convergenceThr:
        transitions, ph = calc_parameters(transitions, ph, fasta.T, ll_history)

    transitions, ph = calc_parameters(transitions, ph, fasta.T, ll_history, plot_mode_only=False)
