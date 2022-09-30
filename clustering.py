"""
G is the set of all vectors
c is the list of vectos where c_i is the group no. to which vector x_i belongs; 0<=i<=n
G_j is the set of all vectors where c_i = j
Z_j is the group representative of group j; 1<=j<=k

J_clust is the mean squared distance of all the vectors from thier respective group representatives

K-means algorithm:

Randomly assign k vectors to different groups 
1) Calculate the closest group representative for each vector, and add to that group
2) Find new group representatives after reallocation, by getting the mean value of each vector
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

EXP = 0.00001
    
def nearest_neighbour(x, Z):
    a = np.array(x)
    return np.argmin([np.linalg.norm(a-np.array(b)) for b in Z])

def assign_groups(x, c, Z):
    G = [list() for i in range(len(Z))]
    for i in range(len(x)):
        j = nearest_neighbour(x[i], Z)
        c[i] = j
        G[c[i]].append(i)
    
    # for i in range(len(G)):
        # if len(G) == 0:
            # G.pop(i)
            
    return G
    
def group_representatives(x, c, G):
    Z = [0]*len(G)
    for j in range(len(G)):
        Z[j] = np.mean([x[i] for i in G[j]], 0)
    return Z

def get_J_clust(x, c, Z):
    J_sum = 0
    for i in range(len(x)):
        J_sum += find_distance(x[i], Z[c[i]])**2
    return J_sum/len(x)
    

def k_means(x, k):
    x_copy = deepcopy(x)
    c = [0]*len(x)
    random.shuffle(x_copy)
    Z = x_copy[:k] # randomly assign k group representatives
    J_clust = None
    while True:
        G = assign_groups(x, c, Z)
        Z = group_representatives(x, c, G)
        new_J_clust = get_J_clust(x, c, Z)
        if J_clust and (J_clust - new_J_clust) <= EXP:
            break
        else:
            J_clust = new_J_clust
    return c, G, Z
    
    
def main():
    data = [(random.randint(0, 100), random.randint(0, 100)) for i in range(100)]
    k = 3
    c, G, Z = k_means(data, k)
    for j in range(len(G)):
        x = []
        for c_i in G[j]:
            x.append(data[c_i])
        res = list(zip(*x))
        plt.scatter(res[0], res[1], label = str(j))
        
    # plot grp representatives
    grp_repr = list(zip(*Z))
    plt.scatter(grp_repr[0], grp_repr[1], label = 'Grp. Repr(s)')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    main()

    