import numpy as np
from matplotlib import pyplot as plt

size = 10
n_agents = 100
deg = 10

pos = np.random.random((n_agents, 2)) * size

plt.plot(pos[:, 0], pos[:, 1], "kx")

for k in range(n_agents):
    each = pos[k]
    dist = np.sum((pos - each) ** 2, axis=1)
    dist[k] = np.inf
    for _ in range(deg):
        i = np.argmin(dist)
        plt.plot([each[0], pos[i, 0]], [each[1], pos[i, 1]])
        dist[i] = np.inf

plt.title(f"{n_agents} agents, {deg} degree connection")
plt.show()
