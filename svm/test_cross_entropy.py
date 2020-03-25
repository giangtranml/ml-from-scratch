import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(p, q):
    return -p*np.log(q) - (1-p)*np.log(1-q)

def mse(p, q):
    return (p - q)**2

p = 0
q = np.linspace(0.0, 1.0)

cost_entropy = cross_entropy(p, q)
cost_mse = mse(p, q)

print(cost_entropy[0])

plt.plot(q, cost_entropy, c='b')
plt.plot(q, cost_mse, c='r')
plt.show()
