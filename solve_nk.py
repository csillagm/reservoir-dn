import numpy as np
import time
import datetime

from parameters import signal_T, dt, training_T, eval_T, align, nk_mapping, topology_mapping, alpha, n_fourier, \
                        period_fourier
from functions.nk_landscape import neighbourmap, randommap, generate_nklandscape, bitstring_to_fitness
from functions.topology import interaction_map_2d_square, interaction_map_1d, interactor
from functions.force_learning import force
from functions.reservoir import generate_population
from functions.signal import generate_fourier_signal, generate_signal, gen_white_noise, gen_fourier_noise, noise_power,\
                    align_signal, fourier_series_from_coeffs, signal_to_bitstring_center
from functions.information_measure import all_values_of_landscape, compute_infomeasure


def force_nk(population_size,
             generations,
             N,
             K,
             nk_mapping,
             topology,
             noise_type,
             noise_amplitude,
             load_population,
             load_initial_conditions,
             load_landscape,
             save_information_measure,
             save_data,
             outfilename_format):

    start_time = time.time()

    eval_length = int(eval_T / dt)
    timebin = eval_length / N

    # store data
    signal_matrix = np.empty([population_size, generations + 1], dtype=list)
    initial_fourier_signals = np.empty(population_size, dtype=list)
    fitness = np.empty([population_size, generations + 1], dtype=float)
    binary_strings = np.empty([population_size, generations + 1], dtype=list)

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

    # Load or generate NK landscape
    if load_landscape != False:
        landscape_file = np.load(load_landscape, allow_pickle=True)
        landscape = landscape_file["nk_landscape"]
        N = landscape_file["N"]
        K = landscape_file["K"]

        print("NK-landscape loaded from file ",load_landscape)
        print()

    else:
        landscape = generate_nklandscape(N, K)


    if nk_mapping == 'neighbour':
        nk_map = neighbourmap(N, K)
    elif nk_mapping == 'random':
        nk_map = randommap(N, K, f=None)
    else:
        assert False, "invalid nk mapping method" + nk_mapping


    # Topology map
    if topology == '1D':
        interaction_map = interaction_map_1d(population_size)
    elif topology == '2D_square':
        interaction_map = interaction_map_2d_square(population_size)
    else:
        assert False, "invalid topology map method" + topology

    # print model parameters
    print("Population size= ", population_size)
    print("Generations= ", generations)
    print("N= ", N, " K= ", K)
    print("NK landscape mapping: ", nk_mapping)
    print("Topology mapping: ", topology)
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

    # simulation
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
                    noise_function = np.zeros(int(signal_T/dt))
                signal_matrix[reservoir, generation] += noise_function


            if align:
                aligned_signal, eval_start_index = align_signal(signal_matrix[reservoir,generation], eval_length,
                                                                timebin)
                bitstring = signal_to_bitstring_center(aligned_signal, timebin, threshold=0)

            else:
                eval_start_index = timebin
                bitstring = signal_to_bitstring_center(signal_matrix[reservoir,generation][timebin:timebin+eval_length],
                                                       timebin, threshold=0)

            # if learner: calculate power of the evaluated part of the noise function, and save it
            if reservoir in learners_list:
                power = noise_power(noise_function[eval_start_index:eval_start_index+eval_length])
                population[reservoir][4][generation - 1].append(power)

            # save binary strings
            binary_strings[reservoir, generation] = np.array(list(bitstring), dtype=int)

            # evaluate fitness
            fitness[reservoir, generation] = bitstring_to_fitness(bitstring, landscape, nk_map, N, K)

        # quit before training at last cycle
        if generation == generations:
            break

        # teacher - learner pairs based on topology
        teacher_learner = interactor(fitness[:,generation], interaction_map)

        # learner's list - in next cycles's signal generation noise is added to these reservoirs' signals
        learners_list = np.empty(population_size, dtype=int)
        copy_id = 0

        # signal copying
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

        print(fitness)

        # print runtime of cycle
        print("Elapsed time:", int(round(time.time() - start_time)/60), "min")

    print("Total time:", int(round(time.time() - start_time)/60), "min")

    if outfilename_format == "datetime":
        # filename = current date and time in population_size-generations-N-K-MMDD-HHMM format
        outfile = str(population_size) + "-" + str(generations) + "-" + str(N) + "-" + str(K) + "-" + \
                  datetime.datetime.now().strftime("%m%d-%H%M")

    elif outfilename_format == "fig3d":
        # filename for fig 3d runs
        outfile = str(N) + "-" + str(K) + "-" + str(load_initial_conditions) + "-" + str(topology_mapping)

    elif outfilename_format == "fig4":
        # filename: noise type - amplitude noise analysis runs
        outfile = str(noise_type)+"-"+str(noise_amplitude)+"-"+datetime.datetime.now().strftime("%m%d-%H%M")

    else: outfile = str(population_size) + "-" + str(generations) + "-" + str(N) + "-" + str(K)


    simulation_params = [population_size, generations, signal_T, eval_T, N, K, nk_mapping, topology_mapping, n_fourier,
                         alpha, align]
    noise_params = [noise_type, noise_amplitude]

    nk_landscape = landscape
    binary_strings = binary_strings
    fitness = fitness
    copying_events = np.empty(population_size, dtype=list)
    for network in range(population_size):
        copying_events[network] = population[network][4]

    if save_information_measure:
        fitness_absolute = fitness
        landscape_values = all_values_of_landscape(nk_landscape, return_sorted=True)
        fitness_measure = compute_infomeasure(fitness_absolute, landscape_values)

        if save_data:
            outfile += "_i"
            np.savez(outfile, sim_params=simulation_params,
                              noise_params=noise_params,
                              nk_landscape=nk_landscape,
                              binary_strings=binary_strings,
                              fitness_abs=fitness_absolute,
                              fitness_measure=fitness_measure,
                              copying_events=copying_events)

            print(outfile, "saved")

            return simulation_params, noise_params, nk_landscape, binary_strings, fitness_absolute, fitness_measure, copying_events

    else:
        if save_data:
            np.savez(outfile,sim_params=simulation_params,
                             noise_params=noise_params,
                             nk_landscape=nk_landscape,
                             binary_strings=binary_strings,
                             fitness=fitness,
                             copying_events=copying_events)

            print(outfile, "saved")

            return simulation_params, noise_params, nk_landscape, binary_strings, fitness, copying_events

    print()

'''
if __name__ == '__main__':
    force_nk(population_size=100,
             generations=200,
             N=20,
             K=1,
             nk_mapping="neighbour",
             topology='1D',
             noise_type="white",
             noise_amplitude=0,
             load_population=False,
             load_initial_conditions=False,
             load_landscape=False,
             save_information_measure=True,
             save_data=True,
             outfilename_format="datetime")
'''

