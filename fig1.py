
# Fig. 1 The training and validation losses of the estimator versus epochs.
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# read data
with open('loss/estimator.json', 'r') as f:
    history_dict = json.load(f)

# plot fig
fig,ax = plt.subplots(figsize=(10,7))
ax.plot(history_dict['loss'], label='Training loss', linewidth=2.0)
if 'val_loss' in history_dict:
    ax.plot(history_dict['val_loss'], label='Validation loss', linewidth=2.0)
plt.xlabel('Epochs', fontsize=26)
plt.ylabel('Loss', fontsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.yscale('log')
plt.xlim([0, 200])
plt.xticks(np.arange(0,201,50))
plt.ylim([0.0001, 0.1])
plt.legend(fontsize=22)
plt.tight_layout()
ax.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig1.pdf')
plt.show()