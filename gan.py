
# generate fake CSI with given location and velocity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from process import CSI, CSI_comp, label_repe
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(CSI_comp, label_repe,
                                                    test_size=0.2, random_state=1)

# set hyperparameters
s = 3
z_dim = 500
batch_size = 128
epoch = 20000
decay_factor = 0.1
patience = 2000

# build generator
def build_generator():
    noise_input = layers.Input(shape=(z_dim,))
    label_input = layers.Input(shape=(3,))
    input = layers.Concatenate()([noise_input, label_input])

    x = layers.Dense(2 * 4 * 2 * 256, input_dim=z_dim+3)(input)
    x = layers.Reshape((2, 4, 2, 256))(x)

    x = layers.UpSampling3D(size=(2, 2, 2))(x) 
    x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.Conv3D(2, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
    x = layers.Cropping3D(cropping=((2, 2), (0, 0), (1, 1)))(x)

    output = layers.Activation('tanh')(x)
    return models.Model([noise_input, label_input], output)

# build discriminator
def build_discriminator():
    input = layers.Input(shape=(Nc, Nt, Ns, 2))

    x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same')(input)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(2, 2, 2),
                       padding='same')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)

    output = layers.Dense(1)(x)
    return models.Model(input, output)

# customized loss
def custom_mse(y_true, y_pred):
    weight = tf.constant([2,1,0.3], dtype=tf.float32)
    error = y_true - y_pred
    squared_difference = tf.square(error * weight)
    loss = tf.reduce_mean(squared_difference)
    return loss

# build model
generator = build_generator()
discriminator = build_discriminator()
estiminator = models.load_model('model/estimator.h5',
                                custom_objects={'custom_mse': custom_mse})

# train
g_losses, d_losses, e_losses, dgan_losses, ggan_losses, gp_losses=[[] for _ in range(6)]
def train(lr = 0.0001):
    g_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.1, beta_2=0.999)
    d_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.1, beta_2=0.999)
    no_improve_steps = 0
    best_dgan_loss = float('inf')
    best_e_loss = float('inf')
    best_i_loss = float('inf')

    # gradient penalty
    def gradient_penalty(discriminator, real_images, fake_images, batch_size):
        alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = discriminator(interpolated)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    for iteration in range(epoch):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        real_labels = y_train[idx].reshape(batch_size, 3)
        z = np.random.normal(0, 1, (batch_size, z_dim))
        fake_labels = np.random.uniform(0, 1, (batch_size, 3))
        fake_images = generator([z, fake_labels])
        
        # train discriminator
        with tf.GradientTape() as d_tape:
            fake_logits = discriminator(fake_images, training=True)
            real_logits = discriminator(real_images, training=True)
            dgan_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp_loss = gradient_penalty(discriminator, real_images,
                                       fake_images, batch_size)
            d_loss = dgan_loss + gp_loss * 10

        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # train generator
        with tf.GradientTape() as g_tape:
            z = np.random.normal(0, 1, (batch_size, z_dim))
            fake_labels = np.random.uniform(0, 1, (batch_size, 3))
            fake_images = generator([z, fake_labels])
            fake_logits = discriminator(fake_images, training=False)
            fake_esti = estiminator(fake_images, training=False)
            weight = tf.constant([1,1,1], dtype=tf.float32)
            e_loss = tf.reduce_mean(tf.square(weight*(fake_labels - fake_esti)))
            ggan_loss = -tf.reduce_mean(fake_logits)
            g_loss = ggan_loss + 1000*e_loss

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        # save loss
        g_losses.append(g_loss.numpy())
        d_losses.append(d_loss.numpy())
        e_losses.append(e_loss.numpy())
        dgan_losses.append(dgan_loss.numpy())
        ggan_losses.append(ggan_loss.numpy())
        gp_losses.append(gp_loss.numpy())

        i_loss = dgan_loss + 1000*e_loss

        if (iteration + 1) >= 5000:
            if np.abs(i_loss) < best_i_loss:
                best_i_loss = np.abs(i_loss)
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if no_improve_steps >= patience and lr > 5e-7:
                lr = lr * decay_factor
                g_optimizer.learning_rate = lr
                d_optimizer.learning_rate = lr
                no_improve_steps = 0
                best_i_loss = np.abs(i_loss)
                print('learning rate changed to:', lr)

            if e_loss < best_e_loss:
                best_e_loss = e_loss
                generator.save('model/best_generator.h5')

        if (iteration + 1) % 1000 == 0:
            # print(f'{iteration + 1} [dgan_loss: {dgan_loss}] [e_loss: {e_loss}]')
            print(f'{iteration + 1} [D loss: {d_loss}] [dgan_loss: {dgan_loss}] [gp_loss: {gp_loss}]')
            print(f'{iteration + 1} [G loss: {g_loss}] [ggan_loss: {ggan_loss}] [e_loss: {e_loss}]')
            # generator.save(f'model/G{iteration + 1}.h5')
            # discriminator.save(f'model/D{iteration + 1}.h5')

# run 
train()

# save model and loss
generator.save('model/generator.h5')
discriminator.save('model/discriminator.h5')

np.save('loss/g_losses.npy', g_losses)
np.save('loss/d_losses.npy', d_losses)
np.save('loss/e_losses.npy', e_losses)
np.save('loss/dgan_losses.npy', dgan_losses)
np.save('loss/ggan_losses.npy', ggan_losses)
np.save('loss/gp_losses.npy', gp_losses)
