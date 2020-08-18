import numpy as np
import random
import time
import datetime

from parameters import dt, signal_T, eval_T, n_fourier, period_fourier
from functions.signal import generate_fourier_signal, align_signal, signal_to_bitstring_center
from functions.tsp import signal_to_permutation



def sample_binaries(n_samples, N, save_to_npz=True):
    """
    Generates samples of N-long binary strings, 1) drawn uniformly, 2) mapped from randomly generated Fourier signals.

    Parameters
    ----------
    n_samples
    N
    save_to_npz

    Returns
    -------
    n_unique_random - the number of different binaries generated drawn uniformly over time
    n_unique_fourier - the number of different binaries mapped from Fourier-signals over time
    occ_distribution - array containing how many binary string were sampled n times using method 2)

    optionally an npz file containing the simulation data

    """
    n_unique_random = np.empty(n_samples, dtype=int)
    n_unique_fourier = np.empty(n_samples, dtype=int)

    occ_distribution = np.zeros(n_samples, dtype=int)

    eval_length = int(eval_T/dt)
    timebin = eval_length/N

    start_time = time.time()


    # random sample from binary space
    random_binaries = np.empty(n_samples, dtype=list)

    unique_random = []

    # Fourier signal -> binary
    fourier_binaries = np.empty(n_samples, dtype=list)

    unique_fourier = []

    for sample in range(n_samples):

        if sample % 10000 == 0:
            print("Elapsed time:", int(round(time.time() - start_time) / 60), "min")
            print("Sample", sample)

        random_binaries[sample] = [random.randint(0, 1) for j in range(N)]

        if random_binaries[sample] not in unique_random:
            unique_random.append(random_binaries[sample])

        n_unique_random[sample] = len(unique_random)

        signal = generate_fourier_signal(n_fourier, period_fourier, signal_T, dt)[0]
        aligned_signal = align_signal(signal, eval_length, timebin)[0]
        fourier_binaries[sample] = signal_to_bitstring_center(aligned_signal, timebin, threshold=0)

        if fourier_binaries[sample] not in unique_fourier:
            unique_fourier.append(fourier_binaries[sample])

        n_unique_fourier[sample] = len(unique_fourier)

    occurences = np.unique(fourier_binaries, return_counts=True)[1]
    for item in occurences:
        occ_distribution[item] = occ_distribution[item] + 1

    print("Elapsed time:", int(round(time.time() - start_time) / 60), "min")

    print(n_unique_random)
    print(n_unique_fourier)

    if save_to_npz:
        # save results
        outfile = "Binary-gen-" + str(n_samples) + "-" + str(N) + "-" + datetime.datetime.now().strftime("%m%d-%H%M")
        np.savez(outfile, sim_parameters=[n_samples, N],
                          random=n_unique_random,
                          fourier=n_unique_fourier,
                          fourier_distribution=occ_distribution)

        print(outfile, "saved")

    print("Total time:", int(round(time.time() - start_time)/60), "min")

    return n_unique_random, n_unique_fourier, occ_distribution



# output file: Perm-gen - *repeat* - *n_samples* - *N* - date-time

# Generates random permutations and permutations mapped from randomly generated Fourier-signals.
# Count the number of unique permutations dynamically at each sample point.
# Sampling happens at a logarithmic scale because we plot on log scale.

# Count the distribution over permutations from Fourier-signals - how many of them we sample 1,2,... times.


N = 10
n_samples = int(1e7)
n_sample_points = int(1e3)

def sample_permutations(n_samples, n_sample_points, N, save_to_npz=True):

    ls_points = np.logspace(0, 7, num=n_sample_points).astype(int)

    numbers = np.arange(1, N+1)

    n_unique_random = np.zeros(n_samples, dtype=int)
    n_unique_fourier = np.zeros(n_samples, dtype=int)
    occ_distribution = np.zeros(n_samples, dtype=int)

    eval_length = int(eval_T/dt)
    timebin = eval_length/N

    start_time = time.time()

    # random sample from permutation space
    random_permutations = np.empty(n_samples, dtype='S10')
    # Fourier signal -> permutation
    fourier_permutations = np.empty(n_samples, dtype='S10')

    # initialize
    random_permutations[0] = "".join(map(str, np.random.permutation(numbers)))

    signal = generate_fourier_signal(n_fourier, period_fourier, signal_T, dt)[0]
    aligned_signal = align_signal(signal, eval_length, timebin)[0]
    fourier_permutations[0] = "".join(map(str, signal_to_permutation(aligned_signal, N)))

    n_unique_random[0] = 1
    n_unique_fourier[0] = 1

    for sample in range(1,n_samples):

        if sample % 100000 == 0:
            print("Elapsed time:", int(round(time.time() - start_time) / 60), "min")
            print("Sample",sample)

        # sample random permutation
        random_permutations[sample] = "".join(map(str, np.random.permutation(numbers)))

        # permutation from signal with method a
        signal = generate_fourier_signal(n_fourier,period_fourier)[0]
        aligned_signal = align_signal(signal, eval_length, timebin)[0]
        fourier_permutations[sample] = "".join(map(str, signal_to_permutation_a(aligned_signal, N)))

        if sample in ls_points:
            n_unique_random[sample] = len(np.unique(random_permutations[:sample+1]))
            n_unique_fourier[sample] = len(np.unique(fourier_permutations[:sample+1]))

        else:
            n_unique_random[sample] = n_unique_random[sample-1]
            n_unique_fourier[sample] = n_unique_fourier[sample-1]

    occurences = np.unique(fourier_permutations, return_counts=True)[1]
    for item in occurences:
        occ_distribution[item] = occ_distribution[item] + 1

    logspace_points = ls_points.astype(int)

    print("Elapsed time:", int(round(time.time() - start_time) / 60), "min")

    if save_to_npz:
        # save results
        outfile = "Perm-gen-" + str(n_samples) + "-" + str(N) + "-" + str(n_sample_points) + "-" + datetime.datetime.now().strftime("%m%d-%H%M")
        np.savez(outfile, sim_parameters=[n_samples, N],
                          random=n_unique_random,
                          fourier=n_unique_fourier,
                          fourier_distribution=occ_distribution,
                          logspace_points=logspace_points)

        print(outfile, "saved")

    print("Total time:", int(round(time.time() - start_time)/60), "min")

    return n_unique_random, n_unique_fourier, occ_distribution, logspace_points

