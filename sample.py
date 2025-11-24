
# sample
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import scipy.io as scio
from tensorflow.keras import models
from process import normalize_CSI, add_noise, cal_nmse

# speed range
samples = np.arange(1,6)
data_label, data_CSI = [], []

# read data
for sample in samples:
    s_data = scio.loadmat(f'data/s{sample}.mat')['data']
    data_label.append(np.array(s_data['label'][0][0]))
    data_CSI.append(np.array(s_data['CSI'][0][0]))

data_label = np.array(data_label)
label1 = np.squeeze(data_label)
data_CSI = np.squeeze(data_CSI)
[N, Nt, Nc, No] = data_CSI.shape
CSI = data_CSI.reshape([N, Nt, Nc, -1, 14])
[N, Nt, Nc, Nf, Ns] = CSI.shape
CSI1 = CSI[:,:,:,0,:].transpose(0,2,1,3)

print("label:", label1)

# save data
np.save('data/label1.npy', label1) # (5, 3)
np.save('data/CSI1.npy', CSI1) # (5, 12, 32, 14)

# normlize label
label = np.load('data/label.npy') # (231, 6, 3)
label_max = label.max(axis=(0, 1))
label_min = label.min(axis=(0, 1))
label_len = label_max - label_min
label_norm = (label1 - label_min) / label_len
CSI_true = normalize_CSI(CSI1)

# LS estimation
SNR = 15
CSI_noise = add_noise(CSI1, SNR)
CSI_norm = normalize_CSI(CSI_noise) # (5, 12, 32, 14)
CSI_comp = np.stack([np.real(CSI_norm), np.imag(CSI_norm)], axis=-1) # (5, 12, 32, 14, 2)

Nc_pos = [2,4,8,10]
CSI_pilot = np.empty([CSI_norm.shape[0], Nt], dtype='complex')
for i in range(CSI_norm.shape[0]):
    for j in range(len(Nc_pos)):
        for k in range(int(Nt/len(Nc_pos))):
            CSI_pilot[i, 8*j+k] = \
            np.mean(CSI_norm[i, Nc_pos[j]:Nc_pos[j]+2, 8*j+k, 3:7])
CSI_esti = CSI_pilot[:,None,:,None].repeat(Nc, axis=1).repeat(Ns, axis=3) # (5, 12, 32, 14)
CSI_esti_comp = np.concatenate([np.real(CSI_esti[:,:,:,:,None]),
                np.imag(CSI_esti[:,:,:,:,None])], axis=-1) # (5, 12, 32, 14, 2)
CSI_pilot = np.concatenate([np.real(CSI_pilot), np.imag(CSI_pilot)], axis=-1) # (5, 64)

generator = models.load_model(f'model/generator.h5')
noise = np.random.normal(0, 1, (CSI_norm.shape[0], 500))
CSI_gen = generator.predict([noise, label_norm]) # (5, 12, 32, 14, 2)
CSI_inte = np.concatenate([CSI_esti_comp, CSI_gen], axis=-1) # (5, 12, 32, 14, 4)

# load models
SNR = 15
channelnet = models.load_model(f'model/channelnet{SNR}.h5')
integrator = models.load_model(f'model/integrator{SNR}.h5')
CGAN = models.load_model(f'model/generator{SNR}.h5')
DNN = models.load_model(f'model/dnn{SNR}.h5')
LSTM = models.load_model(f'model/lstm{SNR}.h5')
UNET = models.load_model(f'model/unet{SNR}.h5')

channelnet_pred = channelnet.predict(CSI_esti_comp)
integrator_pred = integrator.predict(CSI_inte)
CGAN_pred = CGAN.predict([noise, CSI_pilot])
dnn_pred = DNN.predict(CSI_pilot)
LSTM_pred = LSTM.predict(dnn_pred)
UNET_pred = UNET.predict(dnn_pred)

channelnet_comp = channelnet_pred[:,:,:,:,0] + 1j * channelnet_pred[:,:,:,:,1]
integrator_comp = integrator_pred[:,:,:,:,0] + 1j * integrator_pred[:,:,:,:,1]
CGAN_comp = CGAN_pred[:,:,:,:,0] + 1j * CGAN_pred[:,:,:,:,1]
LSTM_comp = LSTM_pred[:,:,:,:,0] + 1j * LSTM_pred[:,:,:,:,1]
UNET_comp = UNET_pred[:,:,:,:,0] + 1j * UNET_pred[:,:,:,:,1]

LS_nmse = cal_nmse(CSI_true, CSI_esti)
channelnet_nmse = cal_nmse(CSI_true, channelnet_comp)
Radiomap_nmse = cal_nmse(CSI_true, integrator_comp)
CGAN_nmse = cal_nmse(CSI_true, CGAN_comp)
LSTM_nmse = cal_nmse(CSI_true, LSTM_comp)
UNET_nmse = cal_nmse(CSI_true, UNET_comp)

print('LS:', LS_nmse)
print('ChannelNet:', channelnet_nmse)
print('CGAN:', CGAN_nmse)
print('DNN+LSTM:', LSTM_nmse)
print('UNET:', UNET_nmse)
print('RadioMap:', Radiomap_nmse)