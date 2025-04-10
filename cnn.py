
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
LS_nmse = np.empty([7, int(Nv*N*Nf*0.2)])
ChannelNet_nmse = np.empty([7, int(Nv*N*Nf*0.2)])
RadioMap_nmse = np.empty([7, int(Nv*N*Nf*0.2)])
best_nmse = np.empty([7, int(Nv*N*Nf*0.2)])

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

    generator = models.load_model('model/generator.h5')
    noise = np.random.normal(0, 1, (CSI_comp.shape[0], 500))
    CSI_gen = generator.predict([noise, label_repe]) # (1.386w, 12, 32, 14, 2)
    CSI_inte = np.concatenate([CSI_esti_comp, CSI_gen], axis=-1) # (1.386w, 12, 32, 14, 4)
    CSI_comb = np.concatenate([CSI_esti_comp, CSI_best], axis=-1) # (1.386w, 12, 32, 14, 4)

    x_train, x_test, y_train, y_test = train_test_split(CSI_esti_comp, CSI_comp,
                                       test_size=0.2, random_state=1)
    x_train1, x_test1, x_train2, x_test2 = train_test_split(CSI_inte, CSI_comb,
                                           test_size=0.2, random_state=1)
    label_train, label_test = train_test_split(label_repe, test_size=0.2, random_state=1)
    
    # build integrator
    def build_integrator(s=3, d=4):
        input = layers.Input(shape=(Nc, Nt, Ns, d))
        x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(1, 1, 1),
                              padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(1, 1, 1),
                          padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(1, 1, 1),
                          padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        output = layers.Conv3D(2, (s, s, s), padding='same', activation='tanh')(x)
        return models.Model(input, output)

    # train
    def train(x_train, y_train, x_test, y_test, model, epoch):
        model.summary()
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best_cnn.h5', monitor='val_loss',
                     verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                    patience=50, min_lr=1e-6, verbose=1)
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=128, verbose=1,
                  validation_split=0.25, callbacks=[checkpoint, reduce_lr])
        model = models.load_model('model/best_cnn.h5')
        model.evaluate(x_test, y_test, verbose = 1)
        return model, history

    # train
    model, history = train(x_train, y_train, x_test, y_test, build_integrator(3,2), 100)
    model.save(f'model/channelnet{SNR}.h5')
    for key in history.history:
        history.history[key] = [float(i) for i in history.history[key]]
    with open(f'loss/channelnet{SNR}.json', 'w') as f:
        json.dump(history.history, f)

    model1, history1 = train(x_train1, y_train, x_test1, y_test, build_integrator(3,4), 500)
    model1.save(f'model/integrator{SNR}.h5')
    for key in history1.history:
        history1.history[key] = [float(i) for i in history1.history[key]]
    with open(f'loss/integrator{SNR}.json', 'w') as f:
        json.dump(history1.history, f)
    
    # test
    model = models.load_model(f'model/channelnet{SNR}.h5')
    y_pred = model.predict(x_test)
    y_pred = y_pred[:,:,:,:,0] + 1j * y_pred[:,:,:,:,1] # (N_test, 12, 32, 14)
    model1 = models.load_model(f'model/integrator{SNR}.h5')
    y_pred1 = model1.predict(x_test1)
    y_pred1 = y_pred1[:,:,:,:,0] + 1j * y_pred1[:,:,:,:,1] # (N_test, 12, 32, 14)

    y_test = y_test[:,:,:,:,0] + 1j * y_test[:,:,:,:,1] # (N_test, 12, 32, 14)
    y_esti = x_test[:,:,:,:,0] + 1j * x_test[:,:,:,:,1] # (N_test, 12, 32, 14)

    LS_nmse[int(SNR/5),:] = cal_nmse(y_test, y_esti)
    ChannelNet_nmse[int(SNR/5),:] = cal_nmse(y_test, y_pred)
    RadioMap_nmse[int(SNR/5),:] = cal_nmse(y_test, y_pred1)

# calculate nmse and SNR
print('LS nmse:', np.round(np.mean(LS_nmse, 1), 4))
print('ChannelNet nmse:', np.round(np.mean(ChannelNet_nmse, 1), 4))
print('RadioMap nmse:', np.round(np.mean(RadioMap_nmse, 1), 4))

# save data
np.save('result/LS_nmse.npy', LS_nmse)
np.save('result/ChannelNet_nmse.npy', ChannelNet_nmse)
np.save('result/RadioMap_nmse.npy', RadioMap_nmse)
