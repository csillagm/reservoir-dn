import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from solve_tsp import force_tsp
from functions import tsp

show = True
save = True

generations_to_vis = [0, 1, 2, 3]
load_data = True

if load_data == False:

    # run permutation sampling
    population_size = 4
    generations = 4
    circle = True
    dist_matrix_path = "../example_files/hungary_distance_10.csv"
    topology = "1D"
    noise_type = "white"
    noise_amplitude = 0
    load_population = False
    load_initial_conditions = False
    save_data = True

    [simulation_params, noise_params, dist_matrix, permutations, cost, \
     copying_events, best_signals, best_aligned_signals] = force_tsp(population_size,
                                                                     generations,
                                                                     circle,
                                                                     dist_matrix_path,
                                                                     topology,
                                                                     noise_type,
                                                                     noise_amplitude,
                                                                     load_population,
                                                                     load_initial_conditions,
                                                                     save_data)

    n_cities = simulation_params[4]

else:
    # alternatively: load data from file
    rundata_filepath = '2D/TSP-100-100-0803-0253.npz'
    rundata_filepath = 'TSP-4-4-0726-1812.npz'

    data = np.load(rundata_filepath, allow_pickle=True)



    simulation_parameters = data['sim_params']
    permutations = data['permutations']
    cost = data['cost'] # population_size X (cycles+1) matrix
    copying_events = data['copying_events']
    best_aligned_signals = data['best_aligned_signals']

    population_size = int(simulation_parameters[0])
    generations = int(simulation_parameters[1])
    n_cities = int(simulation_parameters[4])


rows = 2
columns = len(generations_to_vis)

figure, axs = plt.subplots(rows, columns)

figure.set_size_inches(columns * 6, 10)

# plot signals in first row
for i in range(columns):
    generation = generations_to_vis[i]

    # best permutation of generation #i
    best_perm_i = permutations[np.argmin(cost[:, generation]), generation]
    best_cost_i = min(cost[:, generation])

    aligned = best_aligned_signals[generation]

    axs[0, i].set_title("Generation %i" % (generation + 1), fontsize=40)
    axs[0, i].plot(aligned)

    bin_length = int(len(aligned) / n_cities)
    sample_times = [int(((2 * j + 1) * bin_length) / 2) for j in range(n_cities)]

    colors = np.array([cm.nipy_spectral(x) for x in np.linspace(1, 0, n_cities)])

    for idx, node in enumerate(best_perm_i):
        axs[0, i].hlines(y=aligned[sample_times[node - 1]], xmin=0, xmax=sample_times[node - 1],
                         linestyles='--', linewidth=0.6, color=colors[node - 1], zorder=idx)
        axs[0, i].vlines(x=sample_times[node - 1], ymin=-3, ymax=aligned[sample_times[node - 1]],
                         linestyles='--', linewidth=0.6, color=colors[node - 1], zorder=idx)

    axs[0, i].tick_params(axis='y', direction='in', length=6, labelsize=10)

    axs[0, i].set_yticks(aligned[sample_times])
    # axs[0,i].set_yticklabels(range(1,n_cities+1))
    axs[0, i].set_yticklabels(['⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤'])

    for ticklabel, tickcolor in zip(axs[0, i].get_yticklabels(), colors):
        ticklabel.set_color(tickcolor)

    axs[0, i].set_xticks(sample_times)
    axs[0, i].set_xticklabels(['⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤', '⬤'])

    for ticklabel, tickcolor in zip(axs[0, i].get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # get best path of cycle i
    axs[1, i].set_title("cost = %i" % best_cost_i, fontsize=40)
    axs[1, i] = tsp.draw_graph_hungary_10(permutation=best_perm_i, color=colors, plt_axis=axs[1, i], draw_circle=True)

if save:
    plt.tight_layout()
    plt.savefig(os.path.join('3a.png'), format='png', dpi=200)
if show:
    plt.show()
