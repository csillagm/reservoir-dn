import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm

from parameters import n_fourier, signal_T, dt, eval_T, period_fourier

from functions.signal import generate_fourier_signal, align_signal, signal_to_bitstring_center

from tsp.tsp_evaluation import signal_to_permutation_a

load_signal = False
signal_to_load = None
# signal_to_load = 'aligned_signal.npy'

save_signal = False

N = 10

signal_length = int(signal_T / dt)
eval_length = int(eval_T / dt)
timebin = int(eval_length / N)

if load_signal:
    aligned_signal = np.load(signal_to_load)
else:
    signal = generate_fourier_signal(n_fourier, period_fourier, signal_T, dt)[0]
    aligned_signal = align_signal(signal, eval_length, timebin)[0]

# map the signal to binary
bitstring = signal_to_bitstring_center(aligned_signal, timebin, threshold=0)
binary_string = np.array(list(bitstring), dtype=int)

# map the signal to permutation
permutation = signal_to_permutation_a(aligned_signal, N)

# plot

fig, ax = plt.subplots(tight_layout=True)
fig.set_size_inches(9, 8)
# plt.gcf().subplots_adjust(bottom=0.15)

ax.plot(aligned_signal, 'red', linewidth=3)

ax.plot(np.zeros(len(aligned_signal)), 'black')

plt.ylim((-3.5,5))
ax.set_xlabel('time', fontsize=21, labelpad=10)
ax.set_ylabel('activity', fontsize=21, labelpad=12)

# sample_times = [((2*i+1)/2)*timebin*tau for i in range(N)]
bin_length = int(len(aligned_signal)/N)
sample_times = [int(((2*j+1)*bin_length)/2) for j in range(N)]

colors = np.array([cm.nipy_spectral(x) for x in np.linspace(1, 0, N)])

for idx, node in enumerate(permutation):
    ax.hlines(y=aligned_signal[sample_times[node-1]], xmin=-10, xmax=sample_times[node-1],
              linestyles='--', linewidth=1, color=colors[node-1], zorder=idx)

    ax.vlines(x=sample_times[node-1], ymin=np.amin(aligned_signal)-0.5,
              ymax=aligned_signal[sample_times[node-1]],
              linestyles='--', linewidth=1, color=colors[node-1], zorder=idx)

    ax.vlines(x = sample_times[node-1], ymin=aligned_signal[sample_times[node-1]], ymax=np.amax(aligned_signal)+0.7,
              linestyles='-', color='grey', linewidth=0.9)

for i, x in enumerate(sample_times):
    ax.text(x-4, np.amax(aligned_signal)+1, str(binary_string[i]), color='red', fontsize=21)

ax.text(300, 0, 'threshold', fontsize=20)

ax.tick_params(axis='x', length=0, labelsize=24, pad= -45)
ax.tick_params(axis='y', length=0, labelsize=24, pad= -20)


ax.set_yticks(aligned_signal[sample_times])
ax.set_yticklabels(['⚫']*N)
# axs[0,i].set_yticklabels(range(1,n_cities+1))

for ticklabel, tickcolor in zip(ax.get_yticklabels(), colors):
    ticklabel.set_color(tickcolor)

ax.set_xticks(sample_times)
ax.set_xticklabels(['⚫']*N)

for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors):
    ticklabel.set_color(tickcolor)

sns.despine(left=True, bottom=True, right=True)

if save_signal:
    np.save('aligned_signal',aligned_signal)

plt.savefig(os.path.join("fig2a.png"), format='png', dpi=200, bbox_inches="tight")
plt.show()

