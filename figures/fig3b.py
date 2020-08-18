import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import os

from solve_nk import force_nk

show = True
save = True

load_data = True

if load_data == False:

    population_size = 4
    generations = 4
    N = 20
    K = 3
    nk_mapping = "neighbour"
    topology = "1D"
    noise_type = "white"
    noise_amplitude = 0
    ifm = True

    [simulation_params, noise_params, nk_landscape, binary_strings, fitness_absolute, fitness_measure, copying_events]\
                    = force_nk(population_size,
                               generations,
                               N,
                               K,
                               nk_mapping,
                               topology,
                               noise_type,
                               noise_amplitude,
                               load_population=False,
                               load_initial_conditions=False,
                               load_landscape=False,
                               save_information_measure=True,
                               save_data=True,
                               outfilename_format="datetime")


    fitness = fitness_absolute
    fitness_min = np.amin(fitness)
    fitness_max = np.amax(fitness)




else:
    # alternatively: load data from file
    rundata_filepath = '4-4-20-3-0726-1923_i.npz'

    data = np.load(rundata_filepath, allow_pickle=True)

    simulation_parameters = data['sim_params']
    population_size = int(simulation_parameters[0])
    generations = int(simulation_parameters[1])
    nk_landscape_parameters = data['nk_landscape']
    fitness = data['fitness_abs']
    fitness_min = np.amin(fitness)
    fitness_max = np.amax(fitness)


fig, ax = plt.subplots()
cmap = plt.cm.rainbow

ax = sns.heatmap(fitness.T, ax=ax, cmap=cmap, vmin=fitness_min, vmax=fitness_max, cbar=False, annot=False)

norm = colors.Normalize(vmin=fitness_min, vmax=fitness_max)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

ax.set_xticks([])
ax.set_yticks([])
if save:
    fig.savefig(os.path.join('3b_1.png'), format='png', dpi=50)
if show:
    plt.show()


# Figure 3b/2 - creates
max_value = np.empty(generations + 1, dtype=float)
cnt_max = np.empty(generations + 1, dtype=int)
fraction_max = np.empty(generations + 1, dtype=float)


for gen in range(generations + 1):
    max_value[gen] = np.amax(fitness[:, gen])
    cnt_max[gen] = np.count_nonzero(fitness[:, gen] == max_value[gen])
    fraction_max[gen] = cnt_max[gen] / population_size


    cmap = plt.cm.rainbow
    fitness_min = np.amin(max_value)
    fitness_max = np.amax(max_value)
    print(fitness_max)
    norm = colors.Normalize(vmin=fitness_min, vmax=fitness_max)


plt.figure(figsize=(10, 6))
colormap = cmap(norm(max_value))
plt.scatter(np.arange(generations + 1, step=1), fraction_max, s=80, color=colormap)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
clb = plt.colorbar(sm, pad=0.15)
clb.set_label("fitness", fontsize=18, labelpad=10)
clb.ax.yaxis.set_ticks_position('left')
clb.ax.tick_params(axis="both", labelsize=14)

#plt.tick_params(axis="both", labelsize=14)
#plt.xticks(np.arange(generations + 1, step=int(generations / 10)))
plt.xlabel("generation", fontsize=18, labelpad=10)
plt.ylabel("fraction of networks", fontsize=18, labelpad=10)
plt.ylim(0, 1)

if save:
    plt.tight_layout()
    plt.savefig(os.path.join('3b_2.png'), format='png', dpi=200)
if show:
    plt.show()



