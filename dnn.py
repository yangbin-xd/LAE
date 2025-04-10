
# estimate CSI with received pilots
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from process import CSI, CSI_comp, label_repe, normalize_CSI, add_noise
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

DNN_nmse = np.empty([7, int(Nv*N*Nf*0.2)])

for SNR in np.arange(0, 31, 5):
    # channel estimation with pilots
    CSI_noise = add_noise(CSI, SNR)
    CSI_tran = CSI_noise.transpose(1,0,4,3,2,5) # (6, 200, 10, 12, 32, 14)
    CSI_temp = CSI_tran.reshape([Nv*N, Nf, Nc, Nt, Ns]) # (1200, 10, 12, 32, 14)
    CSI_resp = CSI_temp.reshape([Nv*N*Nf, Nc, Nt, Ns]) # (1.2w, 12, 32, 14)
    CSI_norm = normalize_CSI(CSI_resp) # (1.2w, 12, 32, 14)
    
    Nc_pos = [2,4,8,10]
    CSI_pilot = np.empty([CSI_norm.shape[0], Nt], dtype='complex')
    for i in range(CSI_norm.shape[0]):
        for j in range(len(Nc_pos)):
            for k in range(int(Nt/len(Nc_pos))):
                CSI_pilot[i, 8*j+k] = \
                np.mean(CSI_norm[i, Nc_pos[j]:Nc_pos[j]+2, 8*j+k, 3:7])
    CSI_pilot = np.concatenate([np.real(CSI_pilot), np.imag(CSI_pilot)], axis=-1) # (1.2w, 64)

    x_train, x_test, y_train, y_test = train_test_split(CSI_pilot, CSI_comp,
                                                        test_size=0.2, random_state=1)
    label_train, label_test = train_test_split(label_repe,
                                               test_size=0.2, random_state=1)
    
    # build estimator
    def build_dnn():
        input = layers.Input(shape=(2*Nt,))
        x = layers.Dense(128, activation='relu')(input)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(10752, activation='tanh')(x)
        output = layers.Reshape((12, 32, 14, 2))(x)
        return models.Model(input, output)

    # train
    def train(x_train, y_train, x_test, y_test, model, epoch):
        model.summary()
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best_dnn.h5', monitor='val_loss',
                     verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                    patience=10, min_lr=1e-6, verbose=1)
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=128, verbose=1,
                  validation_split=0.25, callbacks=[checkpoint, reduce_lr])
        model = models.load_model('model/best_dnn.h5')
        model.evaluate(x_test, y_test, verbose = 1)
        return model, history

    # train
    model, history = train(x_train, y_train, x_test, y_test, build_dnn(), 100)
    model.save(f'model/dnn{SNR}.h5')
    for key in history.history:
        history.history[key] = [float(i) for i in history.history[key]]
    with open(f'loss/dnn{SNR}.json', 'w') as f:
        json.dump(history.history, f)

    model = models.load_model(f'model/dnn{SNR}.h5')
    y_pred = model.predict(x_test)
    y_pred = y_pred[:,:,:,:,0] + 1j * y_pred[:,:,:,:,1] # (N_test, 12, 32, 14)
    y_test = y_test[:,:,:,:,0] + 1j * y_test[:,:,:,:,1] # (N_test, 12, 32, 14)

    # calculate nmse
    def cal_nmse(CSI, CSI_esti):
        nmse = np.empty(CSI.shape[0])
        for i in range(CSI.shape[0]):
            mse = np.mean(np.abs(CSI[i,:] - CSI_esti[i,:]) ** 2)
            norm_factor = np.mean(np.abs(CSI[i,:]) ** 2)
            nmse[i] = mse / norm_factor
        return nmse

    DNN_nmse[int(SNR/5),:] = cal_nmse(y_test, y_pred)

# print result vs SNR
print('DNN nmse:', np.round(np.mean(DNN_nmse, 1), 4))
