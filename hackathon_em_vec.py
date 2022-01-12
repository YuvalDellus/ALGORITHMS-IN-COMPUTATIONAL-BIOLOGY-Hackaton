
import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
import matplotlib.pyplot as plt
import itertools

np.set_printoptions(linewidth=400)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def eliminate_zeros(arr):
    return arr[arr[:, 1] != 0]


def eliminate_zeros_add_one(arr):
    arr[:, 0] += 1
    arr[:, 1] += 2
    return arr


def array_to_prob(arr):
    return arr[:, 0] / arr[:, 1]


def smooth(arr):
    ret = np.zeros((arr.shape[0] + 9, 2))
    for i in range(9):
        ret[i:-(9-i), :] += arr
    return (ret / 9)[4:-5]


HIGH = 'High'
LOW = 'Low'


def get_num_low(arr):
    arr = smooth(arr)
    arr = array_to_prob(arr)

    plot_size = 500
    for i in range(0, len((arr)), plot_size):
        x = np.arange(plot_size)
        plt.scatter(x, arr[i: i + plot_size], c='blue', s=1)
        plt.plot(x, arr[i: i + plot_size], c='blue', linewidth=0.2)
        plt.ylim(0, 1)
        plt.show()
        input('cont?')
    low_threshold = 0.3
    high_threshold = 0.7
    state = 'Low'
    transitions = 0
    sum_high, sum_low = 0, 0
    for i in range(arr.shape[0]):
        if arr[i] < low_threshold and state == HIGH:
            transitions += 1
            state = LOW
        if arr[i] > high_threshold:
            state = HIGH
        if state == HIGH:
            sum_high += 1
        if state == LOW:
            sum_low += 1

    print('num of transitions: ' + str(transitions))
    print('prob for transition from low to high: ' + str(1 / (sum_low / transitions)))
    print('prob for transition from high to low: ' + str(1 / (sum_high / transitions)))




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


def calc_ll(hl, lh, emission_table):
    a = Automata(hl, lh)
    sum_ll = []
    for element in itertools.product([0, 1], repeat=emission_table.shape[1]):
        cur = 0
        for i in range(1, emission_table.shape[1]):
            cur += a.transition_probabilities[element[i-1], element[i]]
        for i in range(0, emission_table.shape[1]):
            cur += emission_table[element[i], i]
        sum_ll += [cur]
    return logsumexp(sum_ll)


def plot_results(fasta, ph, pl, likelihood_table):

    x = np.arange(fasta.shape[1])
    plt.plot(x, [ph]*len(x), 'r-')
    plt.plot(x, [pl]*len(x), 'g-')
    positions = np.argmax(likelihood_table, axis=0)
    colors = ['red' if i == 0 else 'green' for i in positions]
    prop = fasta[0, :] / fasta[1, :]
    plt.scatter(x=x, y=prop, c=colors, s=0.4)
    plt.ylim(0, 1)
    plt.show()

    get_num_low(prop)


def calc_parameters(ph, pl, hl, lh, fasta, ll_history, plot_mode=False):
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
    if plot_mode:
        print(hl, lh)
        plot_results(fasta, ph, pl, np.exp(forward_table + backward_table - fasta_llf))
        return

    ll_history += [fasta_llf]
    # phl is probability to transition from H to L
    est_phl_up = np.sum(np.exp(forward_table[0, :-1] + np.log(hl) + emission_table[1, 1:] + backward_table[1, 1:] - fasta_llf))
    est_phl = est_phl_up / np.sum(np.exp(forward_table[0, :-1] + backward_table[0, :-1] - fasta_llf))
    # plh is probability to transition from L to H
    est_plh_up = np.sum(np.exp(forward_table[1, :-1] + np.log(lh) + emission_table[0, 1:] + backward_table[0, 1:] - fasta_llf))
    est_plh = est_plh_up / np.sum(np.exp(forward_table[1, :-1] + backward_table[1, :-1] - fasta_llf))
    f_plus_b = np.exp(forward_table + backward_table - fasta_llf)  # shape = [2, len(seq)]
    p_C = np.multiply(num_C, f_plus_b)
    p_T = np.multiply(num_T, f_plus_b)
    est_ph = np.sum(p_C[0, :]) / (np.sum(p_C[0, :]) + np.sum(p_T[0, :]))
    est_pl = np.sum(p_C[1, :]) / (np.sum(p_C[1, :]) + np.sum(p_T[1, :]))
    print(fasta_llf)
    return est_ph, est_pl, est_phl, est_plh


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)',
                        default="seqs_ATTA.fasta")
    parser.add_argument('--seed', help='Guess for the motif (e.g. ATTA)', default="ATTA")
    parser.add_argument('--ph', type=float, help='Initial guess for the ph transition probability (e.g. 0.01)',
                        default=0.85)
    parser.add_argument('--pl', type=float, help='Initial guess for the pl transition probability (e.g. 0.9)',
                        default=0.2)
    parser.add_argument('--hl', type=float, help='Initial guess for the trans_hl transition probability (e.g. 0.01)',
                        default=0.05)
    parser.add_argument('--lh', type=float, help='Initial guess for the trans_lh transition probability (e.g. 0.9)',
                        default=0.1)
    parser.add_argument('--convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                             ' (e.g. 0.1)', default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    filename = 'Prostate-Epithelial-Z000000S3.beta'
    fasta = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
    fasta = fasta[1166000:1267150, :]
    fasta = eliminate_zeros_add_one(fasta)
    # fasta = eliminate_zeros(fasta)
    get_num_low(fasta)
    # fasta = args.fasta
    ph, pl, hl, lh = args.ph, args.pl, args.hl, args.lh
    # run Baum-Welch
    ll_history = []
    while len(ll_history) < 2 or abs(ll_history[-1] - ll_history[-2]) > args.convergenceThr:
        ph, pl, hl, lh = calc_parameters(ph, pl, hl, lh, fasta.T, ll_history)

    calc_parameters(ph, pl, hl, lh, fasta, ll_history, plot_mode=True)

    # dump results()

# C is mathylated
# T in not

# [x, y] x = num(C)
# y = num(C) + num(T)
