import numpy as np
import matplotlib.pyplot as plt
import os

from functions import representation

save = True
show = True
load_data = True

if load_data == False:

    # run permutation sampling
    n_samples = 20
    n_sample_points = 20
    N = 3

    [random, fourier, frequencies, logspace_points] = representation.sample_permutations(n_samples, n_sample_points, N,
                                                                                         save_to_npz=True)

else:
    # alternatively: load data from file
    rundata_filepath = "../example_files/Binary-gen-3000000-20.npz"

    data = np.load(rundata_filepath)

    parameters = data['sim_parameters']

    n_samples = parameters[0]
    N = parameters[1]

    random = data['random'].reshape(n_samples)
    fourier = data['fourier'].reshape(n_samples)
    frequencies = data['fourier_distribution']


# Figure 2d

plt.figure(figsize=(7, 7))
# plt.title("Average over %d simulations (sample=%d, N= %d) - loglog" %(repeat, n_sample, N))
plt.loglog(range(1, n_samples + 1), fourier, color='blue', linewidth=5)
plt.loglog(range(1, n_samples + 1), random, color='red', linewidth=5)
plt.xlim(left=1)
plt.xlabel("number of solutions", fontsize=20, labelpad=10)
plt.ylabel("number of different solutions", fontsize=20, labelpad=10)
plt.legend(("uniformly sampled in signal space", "uniformly sampled in solution space"), loc="top", frameon=False,
           fontsize=16)
plt.tick_params(axis="both", labelsize=16)
plt.tight_layout()
if save:
    plt.savefig(os.path.join('2d.png'), format='png', dpi=200)
if show:
    plt.show()
plt.close()


# Figure 2e

#set for appropriate layout according to simulation parameters (currently set for n_samples=1e7, n_sample_points=1e4, N=10)
n_bins = 110
ylimits = [1e-3,1e8]

values = np.nonzero(frequencies)[0]
max_value = max(values)
frequencies = frequencies[1:max_value+1]

values = []
for val, freq in enumerate(frequencies):
    for j in range(freq):
        values.append(val + 1)

bins = np.logspace(0, np.log10(max_value+1), n_bins)
widths = (bins[1:] - bins[:-1])

hist = np.histogram(values, bins=bins)
hist_norm = hist[0] / widths

plt.figure(figsize=(7,7))
plt.scatter(bins[:-1], hist_norm)
plt.xscale('log')
plt.yscale('log')
plt.ylim(ylimits)
plt.tick_params(axis="both", labelsize=16)
plt.xlabel("token frequency", fontsize=20)
plt.ylabel("type frequency", fontsize=20)
plt.tight_layout()

if save:
    plt.savefig(os.path.join('2e.png'), format='png', dpi=200)
if show:
    plt.show()

