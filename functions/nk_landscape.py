import numpy as np
import datetime
from random import random, sample


def neighbourmap(N, K):
    nkmap=[]
    for x in range(N-K):
        nkmap.append(list(range(x, x+K)))
    for x in range(N-K, N):
        nkmap.append(list(range(x, N))+list(range(K-N+x)))
    return nkmap


def randommap(N, K, f=None):

    if f==None:
        f=N
    nkmap=[]
    for x in range(f):
        nkmap.append(sample(list(range(N)), k=K))
    return nkmap


def generate_nklandscape(N_nk, K_nk, save_to_npz=False):

    landscape=[]
    for x in range(N_nk):
        landscape.append({})
        for y in range(pow(2, K_nk)):
            landscape[x][y]=random()

    if save_to_npz:

        filename = str(N_nk) + "-" + str(K_nk) + "_landscape_" + datetime.datetime.now().strftime("%H%M")
        np.savez(filename,  N=N_nk,
                            K=K_nk,
                            nk_landscape=landscape)
    return landscape


def bitstring_to_fitness(bitstring, landscape, nkmap, N_nk, K_nk):

    if len(bitstring) != N_nk:
        assert False, "bitsrting should be %i-long" %N_nk

    s=''
    summa=0
    for x in range(N_nk):
        for y in range(K_nk):
            s+=bitstring[nkmap[x][y]]
        summa+=landscape[x][int(s,2)]
        s=''
    return summa / N_nk


def mutate_bitstring(Y, mutation_rate):
    # Y - teacher signal

    N_nk = len(Y)
    randind = (np.where(np.random.random_sample(N_nk, ) < mutation_rate))[0]

    learner = ''
    rind = 0
    for nind in range(N_nk):
        if nind in randind:
            learner = learner + str(Y[randind[rind]]).replace('1', '2').replace('0', '1').replace('2', '0')
            rind += 1 
        else:
            learner = learner + str(Y[nind])

    return learner
        
