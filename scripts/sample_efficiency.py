import matplotlib.pyplot as plt
import numpy as np

SDP = {
    'coffee':
        [54, 66, 64],
    'three_pc':
        [49, 66, 66],
    'square':
        [38, 54, 48],
    'threading':
        [53, 58, 58]}

EDP = {
    'coffee':
        [43, 62, 62],
    'three_pc':
        [15, 50, 70],
    'squre':
        [3, 0, 0],
    'threading':
        [20, 38, 46]}

DP = {
    'coffee':
        [23, 50, 64],
    'three_pc':
        [2, 4, 22],
    'square':
        [0, 2, 2],
    'threading':
        [10, 26, 42]}

plt.figure(figsize=(8, 5))

for data, name, color in zip((SDP, EDP, DP), ('SDP', 'EDP', 'DP'), ('b', 'g', 'r')):
    data = [v for v in data.values()]
    data = np.asarray(data)

    # Compute mean and standard deviation across runs
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)

    # Plot learning curves with shaded region for variance
    plt.plot(np.asarray((100, 316, 1000)), mean_curve, label=name, color=color)
    # plt.fill_between(np.asarray((100, 316, 1000)), mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)

plt.xlabel("# Training Demo")
plt.ylabel("Success Rate")
plt.title("Data Scaling Curves")
plt.ylim(0, 100)
# plt.ylim([10**0, 10**2])
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
