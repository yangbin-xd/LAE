
# Fig. 3 The adversarial losses of the generator and the discriminator versus epochs.
#        The estimate loss and gradient penalty versus epochs.
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

def format_k(x, pos):
    return f'{int(x / 1000)}k' if x >= 1000 else str(int(x))

g_losses = np.load('loss/g_losses.npy')
d_losses = np.load('loss/d_losses.npy')
e_losses = np.load('loss/e_losses.npy')
dgan_losses = np.load('loss/dgan_losses.npy')
ggan_losses = np.load('loss/ggan_losses.npy')
gp_losses = np.load('loss/gp_losses.npy')

font1 = 32
# plot fig31
fig, axs = plt.subplots(figsize=(7, 5))
plt.plot(ggan_losses, label='Adversarial loss of the generator', color='#F65314')
plt.xlabel('Epochs', fontsize=font1, fontweight='bold')
plt.ylabel('Loss', fontsize=font1, fontweight='bold')
# plt.legend(fontsize=22, loc='lower right')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
axs.xaxis.set_major_formatter(FuncFormatter(format_k))
plt.xticks(np.arange(0, 21, 5)*1000)
plt.xlim(0,20000)
plt.yticks(np.arange(-60, 61, 30))
plt.ylim(-60, 60)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.subplots_adjust(left=0.2, right=0.95, top=0.94, bottom=0.20)
plt.savefig('result/fig31.pdf')

# plot fig32
fig, axs = plt.subplots(figsize=(7, 5))
plt.plot(dgan_losses, label='Adversarial loss of the discriminator', color='#FFBB00')
plt.xlabel('Epochs', fontsize=font1, fontweight='bold')
plt.ylabel('Loss', fontsize=font1, fontweight='bold')
# plt.legend(fontsize=22, loc='lower right')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
axs.xaxis.set_major_formatter(FuncFormatter(format_k))
plt.xticks(np.arange(0, 21, 5)*1000)
plt.xlim(0,20000)
plt.ylim(-60, 0)
plt.yticks(np.arange(-60, 1, 20))
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.subplots_adjust(left=0.2, right=0.95, top=0.94, bottom=0.20)
plt.savefig('result/fig32.pdf')

# plot fig33
fig, axs = plt.subplots(figsize=(7, 5))
axs.set_yscale('log')
plt.plot(gp_losses, label='Gradient penalty', color='#7CBB00')
plt.xlabel('Epochs', fontsize=font1, fontweight='bold')
plt.ylabel('Loss', fontsize=font1, fontweight='bold')
# plt.legend(fontsize=22)
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
axs.xaxis.set_major_formatter(FuncFormatter(format_k))
plt.xticks(np.arange(0, 21, 5)*1000)
plt.tight_layout()
plt.xlim(0,20000)
plt.ylim(0.04,3)
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.subplots_adjust(left=0.2, right=0.95, top=0.94, bottom=0.20)
plt.savefig('result/fig33.pdf')

# plot fig34
fig, axs = plt.subplots(figsize=(7, 5))
axs.set_yscale('log')
plt.plot(e_losses, label='Estimate loss', color='#00A1F1')
plt.xlabel('Epochs', fontsize=font1, fontweight='bold')
plt.ylabel('Loss', fontsize=font1, fontweight='bold')
# plt.legend(fontsize=22)
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
axs.xaxis.set_major_formatter(FuncFormatter(format_k))
plt.xticks(np.arange(0, 21, 5)*1000)
# plt.tight_layout()
plt.xlim(0,20000)
plt.ylim(0.001,0.2)
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.subplots_adjust(left=0.2, right=0.95, top=0.94, bottom=0.20)
plt.savefig('result/fig34.pdf')
plt.show()
