# Random modification to test git
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt

def num_permutations(n,k):
    perm = permutations(list(range(n)), k)
    nPk = len(list(perm))
    print(nPk)
    return nPk

def num_combinations (n,k):
    comb = combinations(list(range(n)), k)
    nCk = len(list(comb))
    print(nCk)
    return nCk

def binomial_distribution(p,n,k):
    prob_k = num_combinations(n,k) * (p**k) * ((1-p)**(n-k))
    return prob_k

n = 20
p = 0.5
x_plot = []
y_plot = []

for k in range(n+1):
    prob = binomial_distribution(p,n,k)
    print(prob)
    x_plot.append(k)
    y_plot.append(prob)

plt.plot(x_plot, y_plot, 'ro')
plt.show()