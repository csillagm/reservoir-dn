population_size = 200
generations = 200

# network parameters
n_neurons = 1000
g = 1.5
p = 0.1

# fourier signals
n_fourier = 5
period_fourier = 30
saved_coeffs = 5

# signal noise
noise="white"
fourier_noise_amplitude=0
white_noise_amplitude=0.06

# signal simulation
signal_T = 300 # 300 = optimum for copying quality
dt = 0.1 #timestep

# force learning
alpha = 1 #learning rate for RLS learning rule (1=highest)
training_T = signal_T

# evaluation
eval_T = period_fourier
align = True

# NK landscape
N_nk = 20
K_nk = 5
nk_mapping = 'neighbour' # 'neighbour', 'random'
f = None

# topology
topology_mapping = '1D' # '2D_square', '1D'
