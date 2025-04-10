
# process CSI and label
import numpy as np

# read data
label = np.load('data/label.npy') # (231, 6, 3)
CSI = np.load('data/CSI.npy') # (231, 6, 32, 12, 10, 14)
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

# normlize label to [0,1]
def normalize_label(label):
    label_norm = np.empty(label.shape)
    x_max = np.max(label[:,:,0])
    x_min = np.min(label[:,:,0])
    x_len = x_max - x_min
    label_norm[:,:,0] = (label[:,:,0] - x_min) / x_len
    y_max = np.max(label[:,:,1])
    y_min = np.min(label[:,:,1])
    y_len = y_max - y_min
    label_norm[:,:,1] = (label[:,:,1] - y_min) / y_len
    v_max = np.max(label[:,:,2])
    v_min = np.min(label[:,:,2])
    v_len = v_max - v_min
    label_norm[:,:,2] = (label[:,:,2] - v_min) / v_len
    # print(v_min)
    # print(v_len)
    return label_norm

# normalize CSI
def normalize_CSI(CSI):
    N = CSI.shape[0]
    CSI_norm = np.empty(CSI.shape, dtype='complex')
    for i in range(N):
        max = np.max(np.abs(CSI[i,:]))
        CSI_norm[i,:] = CSI[i,:] / max
    return CSI_norm

# add noise
def add_noise(CSI, SNR):
    power = (np.linalg.norm(CSI))**2/(np.prod(CSI.shape))
    noise = power / (10**(SNR/10))
    np.random.seed(1)
    noise_real = np.random.normal(0, np.sqrt(noise/2), CSI.shape)
    np.random.seed(2)
    noise_imag = np.random.normal(0, np.sqrt(noise/2), CSI.shape)
    noise_comp = noise_real + 1j * noise_imag
    CSI_noise = CSI + noise_comp
    return CSI_noise

# calculate nmse
def cal_nmse(CSI, CSI_esti):
    nmse = np.empty(CSI.shape[0])
    for i in range(CSI.shape[0]):
        mse = np.mean(np.abs(CSI[i,:] - CSI_esti[i,:]) ** 2)
        norm_factor = np.mean(np.abs(CSI[i,:]) ** 2)
        nmse[i] = mse / norm_factor
    return nmse

# process CSI
CSI_noise = add_noise(CSI, 30) # SNR=30
CSI_tran = CSI_noise.transpose(1,0,4,3,2,5) # (6, 231, 10, 12, 32, 14)
CSI_temp = CSI_tran.reshape([Nv*N, Nf, Nc, Nt, Ns]) # (1386, 10, 12, 32, 14)

CSI_resp = CSI_temp.reshape([N*Nv*Nf, Nc, Nt, Ns]) # (13860, 12, 32, 14)
CSI_norm = normalize_CSI(CSI_resp) # (13860, 12, 32, 14)
CSI_comp = np.concatenate([np.real(CSI_norm[:,:,:,:,None]),
           np.imag(CSI_norm[:,:,:,:,None])], axis=-1) # (13860, 12, 32, 14, 2)

# process label
label_norm = normalize_label(label) # (231, 6, 3)
label_tran = label_norm.transpose(1,0,2) # (6, 231, 3)
label_resp = label_tran.reshape([Nv*N, 3]) # (1386, 3)
label_repe = label_resp[:,None,:].repeat(Nf, axis=1).reshape([Nv*N*Nf, -1]) # (13860, 3)

if __name__ == '__main__':
    print(CSI_comp.shape)
    print(label_repe.shape)