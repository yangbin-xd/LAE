
# DNN + LSTM
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from process import CSI, CSI_comp, normalize_CSI, add_noise, cal_nmse
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

# shuffle CSI for upper bound
CSI_copy = CSI_comp.reshape([Nv*N, Nf, Nc, Nt, Ns, 2]).copy()
CSI_copy = CSI_copy.transpose(1,0,2,3,4,5)
np.random.shuffle(CSI_copy)
CSI_copy = CSI_copy.transpose(1,0,2,3,4,5)
CSI_best = CSI_copy.reshape([Nv*N*Nf, Nc, Nt, Ns, 2])

# define result
LSTM_nmse = np.empty([7, int(Nv*N*Nf*0.2)])

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

    CSI_pilot= np.concatenate([np.real(CSI_pilot), 
                               np.imag(CSI_pilot)], axis=-1) # (1.386w, 64)
    DNN = models.load_model(f'model/dnn{SNR}.h5') # (1.2w, 64)
    CSI_esti_comp = DNN.predict(CSI_pilot) # (1.2w, 12, 32, 14, 2)

    # CSI_esti = CSI_pilot[:,None,:,None].repeat(Nc, axis=1).repeat(Ns, axis=3) # (1.2w, 12, 32, 14)
    # CSI_esti_comp = np.concatenate([np.real(CSI_esti[:,:,:,:,None]),
    #                 np.imag(CSI_esti[:,:,:,:,None])], axis=-1) # (1.386w, 12, 32, 14, 2)
    # CSI_esti_comp = CSI_esti_comp[:,:,:,0,:] # (1.386w, 12, 32, 2)
    
    x_train, x_test, y_train, y_test = train_test_split(CSI_esti_comp, CSI_comp,
                                       test_size=0.2, random_state=1)
    
    def build_LSTM():
        input = layers.Input(shape=(Nc, Nt, Ns, 2))
        x = layers.Reshape([Nc*Nt, Ns, 2])(input)
        x = layers.Reshape([Nc*Nt, Ns*2])(x)

        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        
        # # little improvement with the following code
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

        x = layers.Dense(Ns * 2)(x)
        x = layers.Reshape([Nc*Nt, Ns, 2])(x)
        output = layers.Reshape([Nc, Nt, Ns, 2])(x)
        return models.Model(input, output)
    
    # train
    def train(x_train, y_train, x_test, y_test, model, epoch):
        model.summary()
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best_lstm.h5', monitor='val_loss',
                     verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                    patience=10, min_lr=1e-6, verbose=1)
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=128, verbose=1,
                  validation_split=0.25, callbacks=[checkpoint, reduce_lr])
        model = models.load_model('model/best_lstm.h5')
        model.evaluate(x_test, y_test, verbose = 1)
        return model, history

    # train
    model, history = train(x_train, y_train, x_test, y_test, build_LSTM(), 100)
    model.save(f'model/lstm{SNR}.h5')
    for key in history.history:
        history.history[key] = [float(i) for i in history.history[key]]
    with open(f'loss/lstm{SNR}.json', 'w') as f:
        json.dump(history.history, f)

    # test
    model = models.load_model(f'model/lstm{SNR}.h5')
    y_pred = model.predict(x_test)
    y_pred = y_pred[:,:,:,:,0] + 1j * y_pred[:,:,:,:,1] # (N_test, 12, 32, 14)
    y_test = y_test[:,:,:,:,0] + 1j * y_test[:,:,:,:,1] # (N_test, 12, 32, 14)
    LSTM_nmse[int(SNR/5),:] = cal_nmse(y_test, y_pred)

# save nmse and velocity
np.save('result/LSTM_nmse.npy', LSTM_nmse)

# print result vs SNR
print('LSTM nmse:', np.round(np.mean(LSTM_nmse, 1), 4))