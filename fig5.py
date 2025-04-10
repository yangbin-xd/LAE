
# Fig. 10 The NMSEs of the radio map compared with SOTA
#         approaches versus radical velocities.

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from sklearn.model_selection import train_test_split
from process import CSI
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

font1 = 24
font2 = 16

# process velocity
v = np.tile(np.arange(5, 31, 5)[:, np.newaxis], (1, 231))
v = np.tile(v[:, :, np.newaxis], (1, 1, 10))
v = v.reshape([Nv*N, Nf])
v = v.reshape([Nv*N*Nf, -1])
v_train, v_test = train_test_split(v, test_size=0.2, random_state=1)

# load nmse
LS_nmse = np.load('result/LS_nmse.npy')
ChannelNet_nmse = np.load('result/ChannelNet_nmse.npy')
cgan_nmse = np.load('result/cgan_nmse.npy')
LSTM_nmse = np.load('result/LSTM_nmse.npy')
RadioMap_nmse = np.load('result/RadioMap_nmse.npy')

# calculate nmse vs. velocity
velocity_range = np.arange(5,31,5)
LS_nmse1 = np.empty([Nv, 7])
ChannelNet_nmse1 = np.empty([Nv, 7])
cgan_nmse1 = np.empty([Nv, 7])
LSTM_nmse1 = np.empty([Nv, 7])
RadioMap_nmse1 = np.empty([Nv, 7])
v = np.arange(0,31,5)

for i in range(Nv):
    index = np.where(v_test == velocity_range[i])[0]
    LS_nmse1[i,:] = np.mean(LS_nmse[:, index], axis=-1)
    ChannelNet_nmse1[i,:] = np.mean(ChannelNet_nmse[:, index], axis=-1)
    cgan_nmse1[i,:] = np.mean(cgan_nmse[:, index], axis=-1)
    LSTM_nmse1[i,:] = np.mean(LSTM_nmse[:, index], axis=-1)
    RadioMap_nmse1[i,:] = np.mean(RadioMap_nmse[:, index], axis=-1)

# # print result at SNR
# v = 5
# print('LS nmse:', np.round(LS_nmse1[v,:], 4))
# print('ChannelNet nmse:', np.round(ChannelNet_nmse1[v,:], 4))
# print('cgan nmse:', np.round(cgan_nmse1[v,:], 4))
# print('LSTM nmse:', np.round(LSTM_nmse1[v,:], 4))
# print('RadioMap nmse:', np.round(RadioMap_nmse1[v,:], 4))

# plot fig51
LS_nmse1 = np.array([0.4558, 0.283,  0.1554, 0.0855, 0.0577, 0.0506, 0.0507])
ChannelNet_nmse1 = np.array([0.0768, 0.0431, 0.0312, 0.0269, 0.0247, 0.0235, 0.0235])
cgan_nmse1 = np.array([0.0344, 0.0237, 0.0233, 0.0221, 0.0205, 0.0209, 0.021])
LSTM_nmse1 = np.array([0.0343, 0.0225, 0.0179, 0.0157, 0.015,  0.0139, 0.0136])
RadioMap_nmse1 = np.array([0.0549, 0.0288, 0.0184, 0.0144, 0.0115, 0.0116, 0.0108])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(0,31,5))
plt.legend(fontsize=font2)
plt.xlabel('SNR (dB)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.008, 0.6)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig51.pdf')

# plot fig52
LS_nmse1 = np.array([0.4647, 0.2971, 0.1757, 0.1114, 0.0874, 0.0826, 0.084])
ChannelNet_nmse1 = np.array([0.1073, 0.0747, 0.0642, 0.06,   0.0578, 0.0565, 0.0567])
cgan_nmse1 = np.array([0.0629, 0.0532, 0.0462, 0.0458, 0.0427, 0.0413, 0.0428])
LSTM_nmse1 = np.array([0.0583, 0.0423, 0.0378, 0.0342, 0.0343, 0.0326, 0.0293])
RadioMap_nmse1 = np.array([0.0579, 0.0311, 0.0201, 0.0158, 0.0126, 0.0128, 0.0119])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(0,31,5))
plt.legend(fontsize=font2)
plt.xlabel('SNR (dB)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.01, 0.8)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig52.pdf')

# plot fig53
LS_nmse1 = np.array([0.4871, 0.3271, 0.2129, 0.1543, 0.1349, 0.1332, 0.1366])
ChannelNet_nmse1 = np.array([0.1577, 0.1247, 0.114,  0.1101, 0.108,  0.1065, 0.1067])
cgan_nmse1 = np.array([0.109,  0.101,  0.0899, 0.0893, 0.0864, 0.0844, 0.0864])
LSTM_nmse1 = np.array([0.1047, 0.0873, 0.0834, 0.0779, 0.0803, 0.0771, 0.0723])
RadioMap_nmse1 = np.array([0.0668, 0.038,  0.0261, 0.0213, 0.0175, 0.018,  0.0168])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(0,31,5))
plt.legend(fontsize=font2)
plt.xlabel('SNR (dB)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.010, 2)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig53.pdf')
plt.show()



