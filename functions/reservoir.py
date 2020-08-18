import numpy as np
from scipy import sparse

from parameters import n_neurons, signal_T, dt, g, p


def generate_population(pop_size, generations=1):

    population_vector = np.empty(pop_size, dtype=list)
    for i in range(pop_size):
        population_vector[i] = generate_unit(generations)

    return population_vector

def generate_unit(generations=1):

    global p, g, n_reservoir

    Q = sparse.random(n_neurons,n_neurons,dtype='float64',density=p,data_rvs=np.random.randn)*g/np.sqrt(p * n_neurons)
    Q = Q.todense()

    for i in range(0, np.size(Q,1)):
        Q[i,i] = 0

    w_feedback = 2*(np.random.uniform(0, 1, n_neurons) - 0.5)
    w_feedback = w_feedback.reshape([n_neurons, 1])

    w_output = np.random.normal(0, g / np.sqrt(n_neurons * p), n_neurons)

    state = np.random.normal(0, 1, n_neurons)
    copying_events = np.empty(generations, dtype=list)
    reservoir = [Q, w_feedback, w_output, state, copying_events]


    return reservoir

