
# transfer the mat file to npy file
import numpy as np
import scipy.io as scio

# speed range
speeds = np.arange(0,31,5)
data_label, data_CSI = [], []

# read data
for speed in speeds:
    if speed != 0:
        v_data = scio.loadmat(f'data/v{speed}.mat')['data']
        data_label.append(np.array(v_data['label'][0][0]))
        data_CSI.append(np.array(v_data['CSI'][0][0]))

data_label = np.array(data_label)
label = np.transpose(data_label, (1,0,2))
data_CSI = np.squeeze(data_CSI)
CSI = np.transpose(data_CSI, (1,0,2,3,4))

[N, Nv, Nt, Nc, No] = CSI.shape
CSI = CSI.reshape([N, Nv, Nt, Nc, -1, 14])
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

# save data
np.save('data/label.npy', label) # (231, 7, 3)
np.save('data/CSI.npy', CSI) # (231, 7, 32, 12, 10, 14)

print(label.shape)
print(CSI.shape)