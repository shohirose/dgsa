from sklearn_extra.cluster import KMedoids
from dgsa import DGSA
import matplotlib.pyplot as plt
import numpy as np

centers = [[0, 0], [2, 0], [2, 2]]
covs = [[[0.05, 0.0],
         [0.0, 0.2]],
        [[0.1, 0.0],
         [0.0, 0.1]],
        [[0.3, 0.0],
         [0.0, 0.1]]]
rng = np.random.default_rng()
x = None
for center, cov in zip(centers, covs):
    xi = rng.multivariate_normal(
        center,
        cov,
        size=200
    )
    if x is None:
        x = xi
    else:
        x = np.concatenate((x, xi))
km = KMedoids(n_clusters=3).fit(x)
labels = km.labels_
unique_labels = set(labels)
fig = plt.figure()
ax = fig.add_subplot(111)
for label in unique_labels:
    xy = x[labels == label]
    ax.scatter(xy[:, 0], xy[:, 1], label=str(label))
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend()
plt.savefig('samples.png')

dgsa = DGSA(n_bootstraps=3000, seed=0)
std_measures = dgsa.run(x, labels)

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(std_measures.shape[0])
ax.bar(x, std_measures[:, 0], align="edge", label='x', width=-0.3)
ax.bar(x, std_measures[:, 1], align="edge", label='y', width=0.3)
ax.legend()
ax.set_ylabel('Standardized measures of sensitivity')
ax.set_xlabel('Sample')
plt.savefig('test.png')
