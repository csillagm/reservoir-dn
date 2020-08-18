import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt


# converts signal to permutation
# we use one period of the fourier signal
def signal_to_permutation(signal, perm_length):
    '''
    Map countinuous signal to permuatation space

    The signal is split to perm_length intervals of equal length, and evaluated at the middle of these intervals
    The permutation is given by the corresponding sample point ID when taking the descending order of these function values.
    (e.g a monotonically increasing function would result in the identical permutation)

    Parameters
    ----------
    signal : list
        continuous signal to be mapped to permutation
    perm_length : int
        length of permutation, i.e. number of sample points

    Returns
    -------
    permutation : list
    '''

    M = np.empty([2, perm_length])

    # signal binning
    bin_length = int(len(signal) / perm_length)

    for i in range(perm_length):
        M[0, i] = i + 1
        M[1, i] = signal[int(((2 * i + 1) * bin_length) / 2)]

    M_sorted = M[:, M[1, :].argsort()[::-1]]

    perm = np.array([int(x) for x in M_sorted[0, :]])

    return perm


# evaluate length of path given by permutation on graph given by its distance matrix
def evaluate_permutation(distance_matrix, perm, circle):
    '''
    Evaluate length of path represented by a permutation of nodes on graph given by its distance matrix

    Parameters
    ----------
    distance_matrix : array
        (perm_length X perm_length) symmetric, non-negative matrix, where the distance between nodes i and j is given by M[i,j]
    perm : list
        permutation of the numbers [1...perm_length]
    circle : bool
        determines if the salesman travels a closed path or not

    Returns
    cost of the path
    -------

    '''

    cost = 0

    for i in range(len(perm) - 1):
        start = int(perm[i])
        end = int(perm[i + 1])

        cost = cost + distance_matrix[start - 1, end - 1]

    if circle:
        cost = cost + distance_matrix[perm[-1] - 1, perm[0] - 1]

    return cost


# functions for drawing graph of Hungarian cities
def draw_graph_hungary(permutation, color, plt_axis, draw_circle=True):

    G = nx.Graph()
    G.add_node(1, pos=(19.040833, 47.498333))     #Budapest
    G.add_node(2, pos=(21.0877309, 46.6735939))   #Békéscsaba
    G.add_node(3, pos=(21.6273124, 47.5316049))   #Debrecen
    G.add_node(4, pos=(20.3772284, 47.9025348))   #Eger
    G.add_node(5, pos=(17.6503974, 47.6874569))   #Győr
    G.add_node(6, pos=(17.7967639, 46.3593606))   #Kaposvár
    G.add_node(7, pos=(19.6896861, 46.8963711))   #Kecskemét
    G.add_node(8, pos=(20.7784384, 48.1034775))   #Miskolc
    G.add_node(9, pos=(21.7244053, 47.9495324))   #Nyíregyháza
    G.add_node(10, pos=(18.232266, 46.0727345))   #Pécs
    G.add_node(11, pos=(19.7999813, 48.0935237))  #Salgótarján
    G.add_node(12, pos=(20.1414253, 46.2530102))  #Szeged
    G.add_node(13, pos=(18.4221358, 47.1860262))  #Székesfehérvár
    G.add_node(14, pos=(18.7062293, 46.3474326))  #Szekszárd
    G.add_node(15, pos=(20.1824712, 47.1621355))  #Szolnok
    G.add_node(16, pos=(16.6218441, 47.2306851))  #Szombathely
    G.add_node(17, pos=(18.404818, 47.569246))    #Tatabánya
    G.add_node(18, pos=(17.9093019, 47.1028087))  #Veszprém
    G.add_node(19, pos=(16.8416322, 46.8416936))  #Zalaegerszeg

    for i in range(len(permutation)-1):
        G.add_edge(permutation[i],permutation[i+1])

    if draw_circle:
        G.add_edge(permutation[-1],permutation[0])

    pos=nx.get_node_attributes(G,'pos')

    fig = nx.draw(G, pos, node_color=color, with_labels=False, ax=plt_axis)


    # plt.show()

    return fig


def draw_graph_hungary_10(permutation, color, draw_circle, plt_axis):

    G = nx.Graph()
    G.add_node(1, pos=(19.040833, 47.498333))     #Budapest
    G.add_node(2, pos=(21.0877309, 46.6735939))   #Békéscsaba
    G.add_node(3, pos=(21.6273124, 47.5316049))   #Debrecen
    G.add_node(4, pos=(20.3772284, 47.9025348))   #Eger
    G.add_node(5, pos=(17.6503974, 47.6874569))   #Győr
    G.add_node(6, pos=(17.7967639, 46.3593606))   #Kaposvár
    G.add_node(7, pos=(19.6896861, 46.8963711))   #Kecskemét
    G.add_node(8, pos=(20.7784384, 48.1034775))   #Miskolc
    G.add_node(9, pos=(21.7244053, 47.9495324))   #Nyíregyháza
    G.add_node(10, pos=(18.232266, 46.0727345))   #Pécs

    for i in range(len(permutation)-1):
        G.add_edge(permutation[i],permutation[i+1])

    pos=nx.get_node_attributes(G,'pos')

    if draw_circle:
        G.add_edge(permutation[-1],permutation[0])

    fig = nx.draw(G, pos, node_color=color, with_labels=False, ax=plt_axis)

    # nx.draw(G, pos, node_size=10)
    # plt.show()

    return fig


# performs exhaustive search on given distance matrix
def exhaustive_search(distance_matrix):
    nodes = np.array([i for i in range(1, distance_matrix.shape[0] + 1)])
    all_permutations = np.array(list(itertools.permutations(nodes)))

    optimal_solution = None
    optimum = sum(sum(distance_matrix))

    for permutation in all_permutations:
        cost = evaluate_permutation(distance_matrix, permutation, circle=True)
        if cost < optimum:
            optimum = cost
            optimal_solution = permutation

    return optimal_solution, optimum

