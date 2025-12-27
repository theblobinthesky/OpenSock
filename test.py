import numpy as np

r = np.arange(1, 6 + 1)
r2 = np.arange(1, 6 + 1)
sums = r[:, np.newaxis] + r[np.newaxis, :]

pairs = {}
for i in range(2, 12):
    pairs[i] = (sums == i).sum()

total = 36
pairs = {k: float(v / total) for k, v in pairs.items()}
pairs = reversed(sorted(list(pairs.items()), key=lambda x: x[1]))
for k, v in pairs:
    print(f"{k}: {100 * v:.2f}%")