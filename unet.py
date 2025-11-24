
# integrate radio map with pilot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from process import CSI, CSI_comp, label_repe, normalize_CSI, add_noise, cal_nmse
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

# shuffle CSI for upper bound
CSI_copy = CSI_comp.reshape([Nv*N, Nf, Nc, Nt, Ns, 2]).copy()
CSI_copy = CSI_copy.transpose(1,0,2,3,4,5)
np.random.shuffle(CSI_copy)
CSI_copy = CSI_copy.transpose(1,0,2,3,4,5)
CSI_best = CSI_copy.reshape([Nv*N*Nf, Nc, Nt, Ns, 2])

# define result
UNet_nmse = np.empty([7, int(Nv*N*Nf*0.2)])

for SNR in np.arange(0, 31, 5):
    # channel estimation with pilots
    CSI_noise = add_noise(CSI, SNR)
    CSI_tran = CSI_noise.transpose(1,0,4,3,2,5) # (6, 231, 10, 12, 32, 14)
    CSI_temp = CSI_tran.reshape([Nv*N, Nf, Nc, Nt, Ns]) # (1386, 10, 12, 32, 14)
    CSI_resp = CSI_temp.reshape([Nv*N*Nf, Nc, Nt, Ns]) # (1.386w, 12, 32, 14)
    CSI_norm = normalize_CSI(CSI_resp) # (1.386w, 12, 32, 14)

    Nc_pos = [2,4,8,10]
    CSI_pilot = np.empty([CSI_norm.shape[0], Nt], dtype='complex')
    for i in range(CSI_norm.shape[0]):
        for j in range(len(Nc_pos)):
            for k in range(int(Nt/len(Nc_pos))):
                CSI_pilot[i, 8*j+k] = \
                np.mean(CSI_norm[i, Nc_pos[j]:Nc_pos[j]+2, 8*j+k, 3:7])

    CSI_esti = CSI_pilot[:,None,:,None].repeat(Nc, axis=1).repeat(Ns, axis=3) # (1.386w, 12, 32, 14)
    CSI_esti_comp = np.concatenate([np.real(CSI_esti[:,:,:,:,None]),
                    np.imag(CSI_esti[:,:,:,:,None])], axis=-1) # (1.386w, 12, 32, 14, 2)

    x_train, x_test, y_train, y_test = train_test_split(CSI_esti_comp, CSI_comp,
                                       test_size=0.2, random_state=1)
    label_train, label_test = train_test_split(label_repe, test_size=0.2, random_state=1)
    
    # build UNet
    def build_UNet(s=3, d=2):
        input = layers.Input(shape=(Nc, Nt, Ns, d))
        x = layers.ZeroPadding3D(padding=((0, 0), (0, 0), (1, 1)))(input)

        # encoder
        enc1 = layers.Conv3D(32, kernel_size=(s, s, s), padding='same')(x)
        enc1 = layers.BatchNormalization()(enc1)
        enc1 = layers.LeakyReLU(alpha=0.01)(enc1)
        pool1 = layers.MaxPooling3D((2, 2, 2))(enc1)

        enc2 = layers.Conv3D(64, kernel_size=(s, s, s), padding='same')(pool1)
        enc2 = layers.BatchNormalization()(enc2)
        enc2 = layers.LeakyReLU(alpha=0.01)(enc2)
        pool2 = layers.MaxPooling3D((2, 2, 2))(enc2)

        # bottleneck
        bottleneck = layers.Conv3D(128, kernel_size=(s, s, s), padding='same')(pool2)
        bottleneck = layers.BatchNormalization()(bottleneck)
        bottleneck = layers.LeakyReLU(alpha=0.01)(bottleneck)

        # decoder
        up1 = layers.Conv3DTranspose(64, kernel_size=(s, s, s), strides=(2, 2, 2),
                                     padding='same')(bottleneck)
        concat1 = layers.concatenate([up1, enc2])
        dec1 = layers.Conv3D(64, kernel_size=(s, s, s), 
                             padding='same')(concat1)
        dec1 = layers.BatchNormalization()(dec1)
        dec1 = layers.LeakyReLU(alpha=0.01)(dec1)

        up2 = layers.Conv3DTranspose(32, kernel_size=(s, s, s), strides=(2, 2, 2),
                                     padding='same')(dec1)
        concat2 = layers.concatenate([up2, enc1])
        dec2 = layers.Conv3D(32, kernel_size=(s, s, s), 
                             padding='same')(concat2)
        dec2 = layers.BatchNormalization()(dec2)
        dec2 = layers.LeakyReLU(alpha=0.01)(dec2)

        dec2_cropped = layers.Cropping3D(cropping=((0, 0), (0, 0), (1, 1)))(dec2)

        # output
        output = layers.Conv3D(2, kernel_size=(s, s, s), padding='same',
                                activation='tanh')(dec2_cropped)

        return models.Model(input, output)

    # train
    def train(x_train, y_train, x_test, y_test, model, epoch):
        model.summary()
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best_unet.h5', monitor='val_loss',
                     verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                    patience=50, min_lr=1e-6, verbose=1)
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=128, verbose=1,
                  validation_split=0.25, callbacks=[checkpoint, reduce_lr])
        model = models.load_model('model/best_unet.h5')
        model.evaluate(x_test, y_test, verbose = 1)
        return model, history

    # train
    model, history = train(x_train, y_train, x_test, y_test, build_UNet(3,2), 100)
    model.save(f'model/unet{SNR}.h5')
    for key in history.history:
        history.history[key] = [float(i) for i in history.history[key]]
    with open(f'loss/unet{SNR}.json', 'w') as f:
        json.dump(history.history, f)
    
    # test
    model = models.load_model(f'model/unet{SNR}.h5')
    y_pred = model.predict(x_test)
    y_pred = y_pred[:,:,:,:,0] + 1j * y_pred[:,:,:,:,1] # (N_test, 12, 32, 14)
    y_test = y_test[:,:,:,:,0] + 1j * y_test[:,:,:,:,1] # (N_test, 12, 32, 14)
    UNet_nmse[int(SNR/5),:] = cal_nmse(y_test, y_pred)

# calculate nmse and SNR
print('UNet nmse:', np.round(np.mean(UNet_nmse, 1), 4))

# save data
np.save('result/UNet_nmse.npy', UNet_nmse)
