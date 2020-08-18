import numpy as np


def interaction_map_1d(N):
    L = np.empty(N, dtype=list)
    L[0] = [[N-1, 0, 1], 0]
    L[N-1] = [[N-2, N-1, 0], N-1]
    for i in range(1, N-1):
        L[i] = [[i-1, i, i+1], i]

    return L


def interaction_map_2d_square(N):
    a=int(np.sqrt(N))
    m=N%a
    L=[]
    for x in range(N):
        One=[x]
        if x<a:
            if x<m:
                One.append(N-m+x)
            else:
                One.append(N-m-a+x)
        else:
            One.append(x-a)
        if x%a==0:
            if x==N-m:
                One.append(N-1)
            else:
                One.append(x-1+a)
        else:
            One.append(x-1)
        if x%a==a-1:
            One.append(x+1-a)
        elif x==N-1:
            One.append(N-m)
        else:
            One.append(x+1)
        if x>=N-a:
            if x>=N-m:
                One.append(x+m-N)
            else:
                One.append(x+a+m-N)
        else:
            One.append(x+a)
        L.append([set(One),x])
    return L


# inputs: Fitness vector, interaction map
# output: [[teacher, learner]]
def interactor(fitness, L):

    pairs = np.empty(len(fitness), dtype=list)

    for learner in L:
        candidate_teachers = []
        maxfitness = max((fitness[i]) for i in learner[0])
        for i in learner[0]:
            if fitness[i] == maxfitness:
                candidate_teachers.append(i)

        if learner[1] in candidate_teachers:
            teacher = learner[1]
        else:
            teacher = np.random.choice(candidate_teachers)

        pairs[learner[1]] = [teacher, learner[1]]

    return pairs
