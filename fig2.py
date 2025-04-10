
# Fig. 2 The violin plot of the normalized errors for real CSI parameters estimation.
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# load data
error = np.load('result/error.npy')
error = error * np.array([200, 100, 30])
x_labels = ['$x$', '$y$', '$v$']
font1=24

# plot pdf of x
plt.figure(figsize=(7, 5))
kde = gaussian_kde(error[:, 0])
x = np.linspace(min(error[:, 0]), max(error[:, 0]), 1000)
y = kde(x)
plt.plot(x, y, label=x_labels[0], linewidth=2, c='#F65314')
plt.xlabel('Error (m)', fontsize=font1, fontweight='bold')
plt.ylabel('Probability density', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
# plt.legend(fontsize=font1)
plt.xlim([-10,10])
plt.ylim([0,0.16])
plt.xticks(np.arange(-10,11,5))
plt.yticks(np.arange(0,0.16,0.05))
plt.tight_layout()
plt.savefig('result/fig21.pdf')

# plot pdf of x
plt.figure(figsize=(7, 5))
kde = gaussian_kde(error[:, 1])
x = np.linspace(min(error[:, 1]), max(error[:, 1]), 1000)
y = kde(x)
plt.plot(x, y, label=x_labels[1], linewidth=2, c='#FFBB00')
plt.xlabel('Error (m)', fontsize=font1, fontweight='bold')
plt.ylabel('Probability density', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
# plt.legend(fontsize=font1)
plt.xlim([-5,5])
plt.ylim([0,0.6])
plt.xticks(np.arange(-5,6,2.5))
plt.yticks(np.arange(0,0.7,0.2))
plt.tight_layout()
plt.savefig('result/fig22.pdf')

# plot pdf of x
plt.figure(figsize=(7, 5))
kde = gaussian_kde(error[:, 2])
x = np.linspace(min(error[:, 2]), max(error[:, 2]), 1000)
y = kde(x)
plt.plot(x, y, label=x_labels[2], linewidth=2, c='#7CBB00')
plt.xlabel('Error (m/s)', fontsize=font1, fontweight='bold')
plt.ylabel('Probability density', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
# plt.legend(fontsize=font1)
plt.xlim([-2.0,2.0])
plt.ylim([0,1.5])
plt.xticks(np.arange(-2.0,2.1,1))
plt.yticks(np.arange(0,1.6,0.5))
plt.tight_layout()
plt.savefig('result/fig23.pdf')

plt.show()
