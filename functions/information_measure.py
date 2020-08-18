import numpy as np

from functions.nk_landscape import neighbourmap, bitstring_to_fitness


def generate_binary(n, l):
    """
    Generates all possible strings of length n, containing characters in l

    Parameters
    ----------
    n - length of strings
    l - array of strings, a character which are to be used to combine,
        use ['0','1'] for binary strings

    Returns
    -------
    list of all possible strings
    """

    if n == 1:
        return l
    else:
        if len(l) == 0:
            return generate_binary(n-1, ["0", "1"])
        else:
            return generate_binary(n-1, [i + "0" for i in l] + [i + "1" for i in l])



def all_values_of_landscape(landscape, return_sorted=True):
    """
    Calculates each possible value of a given NK-landscape

    Parameters
    ----------
    landscape - array containing the values of component functions in NK landscape
    return_sorted - Boolean, whether to return sorted array of values

    Returns
    -------
    array of all the values
    """

    N = len(landscape)
    K = int(np.log2(len(landscape[0])))

    nkmap = neighbourmap(N, K)

    all_bitstrings = generate_binary(N, ['0', '1'])
    all_values = np.empty(2**N, dtype=float)

    for ind, bitsring in enumerate(all_bitstrings):
        all_values[ind] = bitstring_to_fitness(bitsring, landscape, nkmap, N, K)

    if return_sorted == True:
        all_values.sort()
        return all_values

    elif return_sorted == False:
        return all_values


def compute_infomeasure(fitness, landscape_values):
    '''
    Transform absolute fitness values to information measure.
    Where information(f) = - log_2(p(f)).
    Here p(f) is the probability of randomly sampling a k point on the landscape such that k >= f.

    Parameters
    ----------
    fitness - matrix of absolute fitness values
    landscape_values - array of all values of NK landscape

    Returns
    -------
    information measure matrix, same shape az input fitness matrix
    '''

    landscape_size = len(landscape_values)
    original_shape = fitness.shape
    fitness = fitness.flatten()

    p_f = np.empty(len(fitness), dtype=float)

    for ind, f in enumerate(fitness):
        p_f[ind] = (landscape_size - np.searchsorted(landscape_values,f)) / landscape_size

    info_measure = -np.log2(p_f).reshape(original_shape)

    return info_measure


def attach_info_measure_to_data(filename):

    data = np.load(filename, allow_pickle=True)

    simulation_params = data["sim_params"]
    nk_landscape = data["nk_landscape"]
    binary_strings = data["binary_strings"]
    fitness_absolute = data["fitness"]
    copying_events = data["copying_events"]

    simulation_params_dict = {"population_size": simulation_params[0],
                              "generations": simulation_params[1],
                              "topology": simulation_params[7],
                              "N": simulation_params[4],
                              "K": simulation_params[5],
                              "nk_mapping": simulation_params[6],
                              "signal_T": simulation_params[2],
                              "eval_T": simulation_params[3],
                              "n_fourier": simulation_params[8],
                              "alpha": simulation_params[9],
                              "align": simulation_params[10]}

    if "noise_params" in data.files:
        noise_params = data["noise_params"]

    else: noise_params = ["no noise added"]

    N = int(simulation_params_dict["N"])
    K = int(simulation_params_dict["K"])


    landscape_values = all_values_of_landscape(nk_landscape, return_sorted=True)
    fitness_measure = compute_infomeasure(fitness_absolute, landscape_values)

    outfile = filename.replace(".npz", "_info.npz")

    np.savez(outfile, sim_params=simulation_params,
             sim_params_dict=simulation_params_dict,
             noise_params=noise_params,
             nk_landscape=nk_landscape,
             binary_strings=binary_strings,
             fitness_abs=fitness_absolute,
             fitness_measure=fitness_measure,
             copying_events=copying_events)

    print(outfile, "saved")

