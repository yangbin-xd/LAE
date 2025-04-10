
# estimate the location and velocity based on real CSI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

from process import CSI, CSI_comp, label_repe
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(CSI_comp, label_repe,
                                                    test_size=0.2, random_state=1)

from tensorflow.keras import layers
from tensorflow.keras import models

# estiminator
def build_estimator(s=3, d=2):
    input = layers.Input(shape=(Nc, Nt, Ns, d))
    x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same', activation='relu')(input)
    # x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same', activation='relu')(x)
    # x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same', activation='relu')(x)
    # x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(3, activation='sigmoid')(x)
    return models.Model(input, output)

# customized loss
def custom_mse(y_true, y_pred):
    weight = tf.constant([2,1,0.3], dtype=tf.float32)
    error = y_true - y_pred
    squared_difference = tf.square(error * weight)
    loss = tf.reduce_mean(squared_difference)
    return loss

# train
def train(x_train, y_train, x_test, y_test, model, epoch):
    model.summary()
    model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best_esti.h5', monitor='val_loss',
                 verbose=1, save_best_only=True, mode='min', save_weights_only=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                patience=20, min_lr=1e-6, verbose=1)
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=256, verbose=1,
              validation_split=0.25, callbacks=[checkpoint, reduce_lr])
    model = models.load_model('model/best_esti.h5',
                              custom_objects={'custom_mse': custom_mse})
    model.evaluate(x_test, y_test, verbose = 1)
    return model, history

import json

# train
model, history = train(x_train, y_train, x_test, y_test, build_estimator(), 200)
model.save('model/estimator.h5')
for key in history.history:
    history.history[key] = [float(i) for i in history.history[key]]
with open('loss/estimator.json', 'w') as f:
    json.dump(history.history, f)

# test
model = models.load_model('model/estimator.h5',
                          custom_objects={'custom_mse': custom_mse})
y_pred = model.predict(x_test)
error = y_test - y_pred

# save result
np.save('result/error.npy', error)
print('mean error:', np.mean(error, axis=0))
