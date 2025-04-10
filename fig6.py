
# Fig. 6 The NMSEs of the radio map compared with other approaches
#        versus velocities at different average SNRs.

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
v = np.arange(5,31,5)

for i in range(Nv):
    index = np.where(v_test == velocity_range[i])[0]
    LS_nmse1[i,:] = np.mean(LS_nmse[:, index], axis=-1)
    ChannelNet_nmse1[i,:] = np.mean(ChannelNet_nmse[:, index], axis=-1)
    cgan_nmse1[i,:] = np.mean(cgan_nmse[:, index], axis=-1)
    LSTM_nmse1[i,:] = np.mean(LSTM_nmse[:, index], axis=-1)
    RadioMap_nmse1[i,:] = np.mean(RadioMap_nmse[:, index], axis=-1)

# # print result at SNR
# SNR = 6
# print('LS nmse:', np.round(LS_nmse1[:,SNR], 4))
# print('ChannelNet nmse:', np.round(ChannelNet_nmse1[:,SNR], 4))
# print('cgan nmse:', np.round(cgan_nmse1[:,SNR], 4))
# print('LSTM nmse:', np.round(LSTM_nmse1[:,SNR], 4))
# print('RadioMap nmse:', np.round(RadioMap_nmse1[:,SNR], 4))

# plot fig61
LS_nmse1 = np.array([0.4494, 0.4558, 0.4601, 0.4647, 0.4718, 0.4871])
ChannelNet_nmse1 = np.array([0.0679, 0.0768, 0.0876, 0.1073, 0.1232, 0.1577])
cgan_nmse1 = np.array([0.0285, 0.0344, 0.044,  0.0629, 0.0775, 0.109])
LSTM_nmse1 = np.array([0.0304, 0.0343, 0.0422, 0.0583, 0.0709, 0.1047])
RadioMap_nmse1 = np.array([0.0544, 0.0549, 0.0553, 0.0579, 0.06,   0.0668])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(5,31,5))
plt.legend(fontsize=font2, loc='lower right')
plt.xlabel('Velocity (m/s)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.007, 0.7)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig61.pdf')

# plot fig62
LS_nmse1 = np.array([0.0781, 0.0855, 0.0952, 0.1114, 0.1254, 0.1543])
ChannelNet_nmse1 = np.array([0.0185, 0.0269, 0.0391, 0.06,   0.0763, 0.1101])
cgan_nmse1 = np.array([0.0192, 0.0221, 0.0293, 0.0458, 0.0587, 0.0893])
LSTM_nmse1 = np.array([0.0159, 0.0157, 0.0203, 0.0342, 0.0469, 0.0779])
RadioMap_nmse1 = np.array([0.015,  0.0144, 0.0148, 0.0158, 0.0175, 0.0213])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(5,31,5))
plt.legend(fontsize=font2, loc='upper left')
plt.xlabel('Velocity (m/s)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.007, 0.7)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig62.pdf')
# plt.show()

# plot fig63
LS_nmse1 = np.array([0.0421, 0.0507, 0.0634, 0.084,  0.1014, 0.1366])
ChannelNet_nmse1 = np.array([0.0152, 0.0235, 0.0361, 0.0567, 0.0731, 0.1067])
cgan_nmse1 = np.array([0.0183, 0.021,  0.0275, 0.0428, 0.0556, 0.0864])
LSTM_nmse1 = np.array([0.015,  0.0136, 0.0174, 0.0293, 0.0434, 0.0723])
RadioMap_nmse1 = np.array([0.0113, 0.0108, 0.011,  0.0119, 0.0135, 0.0168])

plt.figure(figsize=(7, 5))
plt.plot(v, LS_nmse1, color='#F65314', marker = '^', markersize=10, lw=2, label='LS')
plt.plot(v, ChannelNet_nmse1, color='#FFBB00', marker = 'o', markersize=10, lw=2, label='ChannelNet [23]')
plt.plot(v, cgan_nmse1, color='#7CBB00', lw=2, marker = 's', markersize=10, label='CGAN [29]')
plt.plot(v, LSTM_nmse1, color='#00A1F1', lw=2, marker = 'd', markersize=10, label='DNN+LSTM [24]')
plt.plot(v, RadioMap_nmse1, color='#68217A', lw=2, marker = 'p', markersize=10, label='Radio Map')

plt.xticks(np.arange(5,31,5))
plt.legend(fontsize=font2)
plt.xlabel('Velocity (m/s)', fontsize=font1, fontweight='bold')
plt.ylabel('NMSE', fontsize=font1, fontweight='bold')
plt.xticks(fontsize=font1, fontweight='bold')
plt.yticks(fontsize=font1, fontweight='bold')
plt.yscale('log')
plt.ylim(0.007, 0.7)
plt.tight_layout()
plt.grid(True, which='both', ls=':', color='gray', alpha=0.3)
plt.savefig('result/fig63.pdf')
plt.show()