import numpy as np
import time
import datetime

from parameters import signal_T, dt, training_T, eval_T, align, nk_mapping, topology_mapping, alpha, n_fourier, \
                        period_fourier
from functions.tsp import signal_to_permutation, evaluate_permutation
from functions.topology import interaction_map_2d_square, interaction_map_1d, interactor
from functions.force_learning import force
from functions.reservoir import generate_population
from functions.signal import generate_fourier_signal, generate_signal, gen_white_noise, gen_fourier_noise, noise_power,\
                    align_signal, fourier_series_from_coeffs, signal_to_bitstring_center


# the salesman travels a closed path or not
circle = True


def force_tsp(population_size,
              generations,
              circle,
              dist_matrix_path,
              topology,
              noise_type,
              noise_amplitude,
              load_population,
              load_initial_conditions,
              save_data):

    start_time = time.time()

    # load distance matrix
    dist_matrix = np.genfromtxt(dist_matrix_path, delimiter=';', dtype=int)
    n_cities = dist_matrix.shape[0]

    eval_length = int(eval_T/dt)

    # store data
    signal_matrix = np.empty([population_size, generations + 1], dtype=list)
    aligned_signal_matrix = np.empty([population_size, generations + 1], dtype=list)
    initial_fourier_signals = np.empty(population_size, dtype=list)
    cost = np.empty([population_size, generations + 1], dtype=float)
    permutations = np.empty([population_size, generations + 1], dtype=list)
    best_signals = np.empty(generations + 1, dtype=list)
    best_aligned_signals = np.empty(generations + 1, dtype=list)

    # Load or generate population of reservoirs
    if load_population != False:
        population_file = np.load(load_population, allow_pickle=True)
        population = population_file['best_units']
        for reservoir in population:
            reservoir[4] = np.empty(generations, dtype=list)

        print("population loaded from file ", load_population)

    else:
        population = generate_population(population_size, generations)
        # reservoir = [M, w_feedback, w_output, state, [[teacher,res], noise_power] ]

    # Topology map
    if topology == '1D':
        interaction_map = interaction_map_1d(population_size)
    elif topology == '2D_square':
        interaction_map = interaction_map_2d_square(population_size)
    else:
        assert False, "invalid topology" + topology

    # print model parameters
    print("Population size= ", population_size)
    print("Generations= ", generations)
    print("Number of cities= ", n_cities)
    print("Topology: ", topology)
    print("Noise type: ", noise_type)
    print("Noise amplitude: ", noise_amplitude)


    # initializing signals
    print()
    print("initializing signals...")
    print()

    if load_initial_conditions != False:
        for network in range(population_size):
            coefficients = np.load(load_initial_conditions, allow_pickle=True)
            initial_fourier_signals[network] = fourier_series_from_coeffs(coefficients[network], signal_T, dt,
                                                                          period_fourier)

        print("initial conditions loaded from file", load_initial_conditions)
        print()

    else:
        for network in range(population_size):
            initial_fourier_signals[network] = generate_fourier_signal(n_fourier, period_fourier, signal_T, dt)

    for reservoir in range(population_size):
        [w_out, state] = force(population[reservoir],initial_fourier_signals[reservoir][0], alpha, training_T, dt)
        population[reservoir][2] = w_out
        population[reservoir][3] = state

    learners_list = []

    print("Elapsed time:", int(round(time.time() - start_time)/60), "min")


    for generation in range(generations + 1):

        print("Generation:", generation)

        # Generate signals and evaluate fitness
        for reservoir in range(population_size):
            signal, state = generate_signal(population[reservoir], signal_T, dt)
            signal_matrix[reservoir, generation] = signal
            population[reservoir][3] = state

            # add noise to learners signal (ie noisy copying)
            if reservoir in learners_list:
                if noise_type == "white":
                    noise_function = gen_white_noise(noise_amplitude, int(signal_T / dt))
                elif noise_type == "fourier":
                    noise_function = gen_fourier_noise(noise_amplitude, signal_T, dt, n_fourier, period_fourier)
                else:
                    noise_function = np.zeros(int(signal_T / dt))
                signal_matrix[reservoir, generation] += noise_function

            if align:
                signal_for_eval, eval_start_index = align_signal(signal_matrix[reservoir, generation], eval_length, timebin=0)


            else:
                eval_start = 0
                signal_for_eval = signal_matrix[network, generation][eval_start: eval_start + eval_length]


            # if learner: calculate power of the evaluated part of the noise function, and save it
            if reservoir in learners_list:
                power = noise_power(noise_function[eval_start_index:eval_start_index + eval_length])
                population[network][4][generation - 1].append(power)

            # save aligned signal
            aligned_signal_matrix[reservoir, generation] = signal_for_eval

            perm = signal_to_permutation(signal_for_eval, n_cities)
            permutations[reservoir, generation] = np.array(list(perm), dtype=int)
            cost[reservoir, generation] = evaluate_permutation(dist_matrix, perm, circle)


        print("Min cost = ", min(cost[:, generation]))
        # save current best signal
        best_signals[generation] = signal_matrix[np.argmin(cost[:, generation]), generation]
        best_aligned_signals[generation] = aligned_signal_matrix[np.argmin(cost[:, generation]), generation]

        # quit before training at last cycle
        if generation == generations:
            break

        # teacher -  learner pairs based on topology
        teacher_learner = interactor((-1) * cost[:, generation], interaction_map)

        # learner's list, in next cycles's signal generation we will add noise to these reservoirs' signals
        learners_list = np.empty(population_size, dtype=int)
        copy_id = 0

        # teaching / copying
        for pair in teacher_learner:
            teacher = pair[0]
            learner = pair[1]
            if learner != teacher:
                [w_out, state] = force(population[learner], signal_matrix[teacher, generation], alpha, training_T, dt)
                population[learner][2] = w_out
                population[learner][3] = state

                learners_list[copy_id] = learner
                copy_id += 1

            # store the copying event:
            population[learner][4][generation] = pair

        learners_list = learners_list[:copy_id]

        # print runtime of cycle
        print("Elapsed time:", int(round(time.time() - start_time)/60), "min")



    # save data to file (filename = current date and time in MMDD-HHMM format)

    simulation_params = [population_size, generations, signal_T, eval_T, n_cities, nk_mapping, topology, n_fourier,
                         alpha, align]
    noise_params = [noise_type, noise_amplitude]

    copying_events = np.empty(population_size, dtype=list)
    for network in range(population_size):
        copying_events[network] = population[network][4]

    if save_data:
        outfile = "TSP" + "-" + str(population_size) + "-" + str(generations) + "-" + datetime.datetime.now().strftime(
            "%m%d-%H%M")

        np.savez(outfile, sim_params=simulation_params,
                          noise_params=noise_params,
                          distance_matrix=dist_matrix,
                          permutations=permutations,
                          cost=cost,
                          copying_events=copying_events,
                          best_signals=best_signals,
                          best_aligned_signals=best_aligned_signals)

        print(outfile," saved")

    print("Total time:", int(round(time.time() - start_time)/60), "min")

    return simulation_params, noise_params, dist_matrix, permutations, cost, copying_events, best_signals, best_aligned_signals

'''
force_tsp(population_size=2,
          generations=5,
              circle=True,
              dist_matrix_path="example_files/hungary_distance_10.csv",
             topology='1D',
              noise_type="white",
              noise_amplitude=0,
              load_population=False,
              load_initial_conditions=False,
              save_data=False)
'''