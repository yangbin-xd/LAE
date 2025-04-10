
# Fig.8 The training and validation losses of the integrator
# versus epochs under different average SNRs.
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# plot training loss and validation loss vs. epoch
fig,ax = plt.subplots(figsize=(11,7.7))
SNR = np.arange(0,30+1,5)
color = ['#F65314', '#FFBB00', '#FFEC40', '#7CBB00', '#00CCCC', '#00A1F1', '#68217A']

for i in range(SNR.shape[0]):
    with open(f'loss/integrator{SNR[i]}.json', 'r') as f:
        loss_dict = json.load(f)
    ax.plot(loss_dict['loss'], c=color[i], linestyle = '-', linewidth=2,
            label=f'{SNR[i]}dB training loss')
    if 'val_loss' in loss_dict:
        ax.plot(loss_dict['val_loss'], c=color[i], linestyle = '--', linewidth=2,
                label=f'{SNR[i]}dB validation loss')

# figure set
plt.xlabel('Epochs', fontsize=26)
plt.ylabel('Loss', fontsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(loc='upper right', fontsize=20, ncol=1, bbox_to_anchor=(1.7, 1))
plt.subplots_adjust(right=0.6)
ax.grid(True, ls=':', color='gray', alpha=0.3)
plt.yscale('log')
plt.xlim(0,500)
plt.xticks(np.arange(0, 501, 100))
plt.ylim(0.0025,0.10)
plt.grid(True, which='both', linestyle=':', color='gray', alpha=0.3)
# plt.tight_layout()

# save figure
plt.savefig('result/fig4.pdf')
plt.show()
